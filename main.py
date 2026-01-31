import os
import uuid
import shutil
import subprocess
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# 配置路径
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
STATIC_DIR = BASE_DIR / "static"
PREVIEW_DIR = STATIC_DIR / "previews"

# 允许的视频扩展名
ALLOWED_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".mpeg", ".mpg", ".flv", ".ts", ".m4v"}

# 确保目录存在
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 内存中存储任务状态（生产环境建议使用数据库）
class JobStatus(BaseModel):
    id: str
    inputs: List[str]
    outputs: List[str]
    params: Dict[str, Any]
    status: str  # running, completed, failed
    command: Optional[str] = None
    error: Optional[str] = None
    input_size: Optional[int] = None
    output_size: Optional[int] = None
    compression_ratio: Optional[float] = None
    duration: Optional[float] = None
    progress: float = 0.0
    created_at: float = Field(default_factory=lambda: datetime.now().timestamp())
    completed_at: Optional[float] = None

JOBS: Dict[str, JobStatus] = {}
JOB_PROCESSES: Dict[str, subprocess.Popen] = {}

class TranscodeParams(BaseModel):
    vcodec: Optional[str] = "libx264"
    acodec: Optional[str] = "aac"
    bitrate: Optional[str] = None
    crf: Optional[int] = None
    preset: Optional[str] = None
    resolution: Optional[str] = None
    format: Optional[str] = "mp4"
    hw_accel: Optional[str] = None
    threads: Optional[int] = 0
    scodec: Optional[str] = "copy" # 字幕编码，默认复制，预览时可设为 None 禁用
    deinterlace: bool = False
    extra_args: Optional[List[str]] = None

class TranscodeRequest(BaseModel):
    inputs: List[str]
    params: TranscodeParams
    output_dir: Optional[str] = None

def get_video_duration(input_path: Path) -> float:
    """使用 ffprobe 获取视频时长(秒)"""
    try:
        cmd = [
            "ffprobe", 
            "-v", "error", 
            "-show_entries", "format=duration", 
            "-of", "default=noprint_wrappers=1:nokey=1", 
            str(input_path)
        ]
        ret = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if ret.returncode == 0:
            return float(ret.stdout.strip())
    except:
        pass
    return 0.0

# 核心转码逻辑
def build_ffmpeg_cmd(input_path: Path, output_path: Path, params: TranscodeParams, input_options: List[str] = None) -> List[str]:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "info"]

    # 硬件加速配置 (必须在 -i 之前)
    use_cuda = False
    use_qsv = False
    use_vaapi = False
    
    if params.hw_accel and params.hw_accel.startswith("cuda"):
        use_cuda = True
        cmd.extend(["-hwaccel", "cuda"])
        cmd.extend(["-hwaccel_output_format", "cuda"])
        
        # 指定 GPU 设备
        if ":" in params.hw_accel:
            device_idx = params.hw_accel.split(":")[1]
            cmd.extend(["-hwaccel_device", device_idx])
            
    elif params.hw_accel == "qsv":
        use_qsv = True
        cmd.extend(["-hwaccel", "qsv"])
        cmd.extend(["-hwaccel_output_format", "qsv"])
        
    elif params.hw_accel == "vaapi":
        use_vaapi = True
        # AMD/Intel VAAPI 通用配置
        cmd.extend(["-hwaccel", "vaapi"])
        cmd.extend(["-hwaccel_device", "/dev/dri/renderD128"]) # 默认渲染设备
        cmd.extend(["-hwaccel_output_format", "vaapi"])

    if input_options:
        cmd.extend(input_options)

    cmd.extend(["-i", str(input_path)])
    
    # 关键：映射所有流 (视频/音频/字幕)
    cmd.extend(["-map", "0"])

    # 视频编码器自动切换
    vcodec = params.vcodec
    if use_cuda:
        if vcodec == "libx264":
            vcodec = "h264_nvenc"
        elif vcodec == "libx265":
            vcodec = "hevc_nvenc"
            
    if vcodec:
        cmd.extend(["-c:v", vcodec])
        
    if params.acodec:
        cmd.extend(["-c:a", params.acodec])
    
    # 字幕处理
    if params.scodec and params.scodec.lower() != "none":
        cmd.extend(["-c:s", params.scodec])
    else:
        # 显式禁用字幕
        cmd.extend(["-sn"])

    if params.bitrate:
        cmd.extend(["-b:v", params.bitrate])
        
    # CRF (注意: nvenc 也支持 -cq/-rc 等，但简单的 -crf 可能被忽略或需要改用 -cq，这里暂且保留，ffmpeg 通常会做适配或忽略)
    # 对于 nvenc，通常用 -rc constqp -qp N 或 -rc vbr -cq N
    # 简单起见，如果使用 nvenc 且指定了 crf，我们尝试保留原样，或者警告。
    # 实际上 ffmpeg 的 h264_nvenc 不支持 -crf，它使用 -cq (VBR) 或 -qp (CQP)
    # 为了简化，如果检测到 nvenc 且有 crf，我们尝试转换为 -cq
    if params.crf is not None:
        if use_cuda and "nvenc" in vcodec:
            cmd.extend(["-rc", "vbr", "-cq", str(params.crf), "-qmin", str(params.crf), "-qmax", str(params.crf)])
        elif use_qsv and "qsv" in vcodec:
             # QSV 通常使用 -global_quality (ICQ 模式)
             cmd.extend(["-global_quality", str(params.crf)])
        else:
            cmd.extend(["-crf", str(params.crf)])
            
    if params.preset:
        cmd.extend(["-preset", params.preset])
        
    # 视频过滤器链 (Scale, Deinterlace)
    filters = []
    
    # 反交错 (建议在缩放前处理)
    if params.deinterlace:
        if use_cuda:
            # CUDA 硬件反交错 (需要 ffmpeg 支持 yadif_cuda)
            # 0:-1:0 -> mode:parity:deint (default)
            filters.append("yadif_cuda=0:-1:0")
        else:
            # CPU 软件反交错
            filters.append("yadif")

    if params.resolution:
        if use_cuda:
            # 使用 scale_cuda 过滤器
            filters.append(f"scale_cuda={params.resolution}")
        else:
            filters.append(f"scale={params.resolution}")
    
    if filters:
        cmd.extend(["-vf", ",".join(filters)])
            
    if params.threads is not None and params.threads > 0:
        cmd.extend(["-threads", str(params.threads)])
    
    if params.extra_args:
        cmd.extend(params.extra_args)
        
    cmd.append(str(output_path))
    return cmd

def run_transcode_job(job_id: str):
    job = JOBS[job_id]
    try:
        # 简单实现：顺序处理所有输入文件
        for idx, input_file in enumerate(job.inputs):
            # 检查任务是否已被取消
            if job.status == "cancelled":
                break

            inp = Path(input_file)
            if not inp.exists():
                raise FileNotFoundError(f"输入文件不存在: {inp}")
            
            # 确定输出路径
            suffix = f".{job.params.get('format', 'mp4')}"
            out_name = inp.stem + suffix
            # 如果指定了输出目录则使用，否则默认
            if job.outputs and len(job.outputs) > idx:
                 # 此时 outputs 已经在创建任务时预填充了，这里确认一下
                 pass
            
            # 这里简单起见，重新计算输出路径以确保正确
            out_dir = OUTPUT_DIR
            if job.outputs and len(job.outputs) > 0:
                 out_dir = Path(job.outputs[0]).parent
            
            output_path = out_dir / out_name
            
            # 构建参数对象
            params_obj = TranscodeParams(**job.params)
            cmd = build_ffmpeg_cmd(inp, output_path, params_obj)
            
            # 更新任务信息中的命令（仅记录最后一条）
            job.command = " ".join(cmd)
            
            # 执行命令
            # 使用 -progress pipe:1 将进度信息输出到 stdout，方便解析
            # 或者直接读取 stderr (ffmpeg 默认输出到 stderr)
            # 为了稳妥，我们读取 stderr，并使用 Universal Newlines
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
            # 存储进程对象
            JOB_PROCESSES[job_id] = process
            
            # 实时读取输出以更新进度
            # FFmpeg 输出到 stderr
            while True:
                line = process.stderr.readline()
                if not line and process.poll() is not None:
                    break
                
                if line:
                    # 尝试解析时间 time=00:00:00.00
                    # 示例: frame=  200 fps= 45 q=28.0 size=    1024kB time=00:00:10.50 bitrate= 799.0kbits/s speed=2.3x
                    if "time=" in line:
                        try:
                            match = re.search(r'time=(\d{2}):(\d{2}):(\d{2}\.\d+)', line)
                            if match:
                                h, m, s = map(float, match.groups())
                                current_seconds = h * 3600 + m * 60 + s
                                if job.duration and job.duration > 0:
                                    progress = min(100.0, (current_seconds / job.duration) * 100)
                                    job.progress = progress
                        except:
                            pass
            
            # 等待结束
            stdout, stderr = process.communicate() # 获取剩余输出
            
            # 清理进程引用
            if job_id in JOB_PROCESSES:
                del JOB_PROCESSES[job_id]
            
            # 再次检查状态
            if job.status == "cancelled":
                break

            if process.returncode != 0:
                # 如果是正常结束但有 stderr 输出是正常的，我们需要区分是否真的出错
                # 通常 returncode != 0 才是真的错
                raise RuntimeError(f"FFmpeg Error: Return Code {process.returncode}")
        
        # 只有未取消且无错误才标记为完成
        if job.status != "cancelled":
            job.status = "completed"
            job.progress = 100.0 # 确保显示完成
            job.completed_at = datetime.now().timestamp()
            
            # 计算输出大小和压缩率
            try:
                # 假设只有一个输出文件（因为我们拆分了任务）
                if job.outputs:
                    out_p = Path(job.outputs[0])
                    if out_p.exists():
                        job.output_size = out_p.stat().st_size
                        
                        if job.input_size and job.input_size > 0:
                            # 压缩率 = 输出 / 输入 (例如 0.5 表示 50%)
                            # 或者用户想要的可能是压缩比例 (Saved space?)
                            # 通常压缩比例指的是 Output Size / Input Size * 100%
                            # 或者 Compression Ratio 2:1 etc.
                            # 这里存储小数比率，前端去格式化
                            job.compression_ratio = job.output_size / job.input_size
            except Exception as e:
                print(f"Error calculating stats: {e}")
                
    except Exception as e:
        # 如果是取消导致的错误，不标记为失败
        if job.status != "cancelled":
            job.status = "failed"
            job.error = str(e)

# API 接口

@app.get("/")
def read_root():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/jobs")
def get_jobs():
    # 按时间倒序返回
    return list(reversed(list(JOBS.values())))

@app.get("/hardware-info")
def get_hardware_info():
    """检测可用硬件加速"""
    info = {
        "cpu": True,
        "cpu_count": os.cpu_count() or 4,
        "cuda": False,
        "cuda_devices": [],
        "qsv": False,
        "vaapi": False
    }
    
    # 1. 检测 NVIDIA CUDA
    # 简单检查 nvidia-smi 是否可用且能返回成功
    if shutil.which("nvidia-smi"):
        try:
            # 运行 nvidia-smi -L 快速检查
            # Output example: GPU 0: NVIDIA GeForce RTX 3060 (UUID: GPU-...)
            ret = subprocess.run(["nvidia-smi", "-L"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            if ret.returncode == 0:
                info["cuda"] = True
                # 解析具体型号
                for line in ret.stdout.strip().split('\n'):
                    if not line.strip(): continue
                    # 匹配 GPU 0: Name (UUID...
                    match = re.search(r'GPU\s+(\d+):\s+(.+?)\s+\(UUID', line)
                    if match:
                        info["cuda_devices"].append({
                            "index": int(match.group(1)),
                            "name": match.group(2)
                        })
                    else:
                        # Fallback parsing if regex fails
                        parts = line.split(':')
                        if len(parts) >= 2:
                            info["cuda_devices"].append({
                                "index": len(info["cuda_devices"]), 
                                "name": parts[1].split('(')[0].strip()
                            })
        except:
            pass
            
    # 2. 检测 Intel/AMD 核显 (VAAPI/QSV)
    # 检查 /dev/dri 目录是否存在
    if Path("/dev/dri").exists():
        # 通常 /dev/dri/renderD128 存在即意味着支持 VAAPI/QSV
        if list(Path("/dev/dri").glob("renderD*")):
            info["qsv"] = True
            info["vaapi"] = True
            
    return info

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail="不支持的文件类型")
    
    file_path = INPUT_DIR / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return {"filename": file.filename, "path": str(file_path)}

@app.post("/scan-directory")
def scan_directory(path: str = Form(...)):
    base_path = Path(path)
    if not base_path.exists():
        raise HTTPException(status_code=400, detail="路径不存在")
    if not base_path.is_dir():
        raise HTTPException(status_code=400, detail="该路径不是目录")
        
    found_files = []
    # 递归扫描
    for p in base_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
            found_files.append(str(p))
            
    if not found_files:
        raise HTTPException(status_code=400, detail="未找到视频文件")
        
    return {"count": len(found_files), "files": found_files}

@app.post("/transcode")
def create_transcode_job(req: TranscodeRequest, background_tasks: BackgroundTasks):
    # 验证输入
    if not req.inputs:
        raise HTTPException(status_code=400, detail="没有输入文件")
        
    created_jobs = []
    
    # 预计算输出路径用于展示
    out_dir = Path(req.output_dir) if req.output_dir else OUTPUT_DIR
    suffix = f".{req.params.format}"
    
    # 将每个输入文件拆分为独立任务
    for inp in req.inputs:
        job_id = uuid.uuid4().hex
        p = Path(inp)
        
        # 获取源文件大小
        input_size = 0
        duration = 0.0
        try:
            if p.exists():
                input_size = p.stat().st_size
                duration = get_video_duration(p)
        except:
            pass
        
        # 单个任务的输出列表
        single_output = [str(out_dir / (p.stem + suffix))]
        
        job = JobStatus(
            id=job_id,
            inputs=[inp], # 只有这一个文件
            outputs=single_output,
            params=req.params.model_dump(),
            status="running",
            input_size=input_size,
            duration=duration,
            progress=0.0
        )
        
        JOBS[job_id] = job
        background_tasks.add_task(run_transcode_job, job_id)
        created_jobs.append(job_id)
    
    # 返回第一个 job_id 兼容旧前端，或者可以返回列表（前端需要适配）
    # 为了兼容现有前端（只接收一个 job_id），我们返回最后一个创建的 ID，
    # 但前端最好能刷新整个列表。
    # 实际上，现在的返回值前端并没有特别依赖 job_id 做跳转，而是刷新列表。
    # 我们返回 "created_count" 让前端知道创建了多少个。
    
    return {"job_id": created_jobs[-1], "status": "running", "created_count": len(created_jobs)}

@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    job = JOBS[job_id]
    if job.status in ["completed", "failed", "cancelled"]:
        return {"status": job.status, "message": "任务已结束"}
        
    # 标记为已取消
    job.status = "cancelled"
    
    # 终止进程
    if job_id in JOB_PROCESSES:
        process = JOB_PROCESSES[job_id]
        try:
            process.terminate() # 尝试优雅终止
            # 也可以选择 kill() 强制终止
        except Exception as e:
            print(f"Error terminating process: {e}")
            
    return {"status": "cancelled", "message": "任务已中止"}

@app.post("/preview")
def create_preview(req: TranscodeRequest):
    if not req.inputs:
        raise HTTPException(status_code=400, detail="没有输入文件")
        
    input_path = Path(req.inputs[0])
    if not input_path.exists():
        raise HTTPException(status_code=400, detail="输入文件不存在")
        
    # 获取时长并计算中间点
    duration = get_video_duration(input_path)
    start_time = max(0, duration / 2 - 5) # 从中间开始，或者至少0
    
    # 生成预览文件名
    preview_filename = f"preview_{uuid.uuid4().hex}.mp4"
    preview_path = PREVIEW_DIR / preview_filename
    
    # 强制 mp4 格式用于 Web 预览
    # 注意：如果用户选了 mkv/mov 等，预览时最好也转为 mp4 以便浏览器播放
    # 但为了真实反映参数效果，我们应尽量保持视频编码参数，只改变封装格式?
    # 或者如果浏览器不支持该编码(如 hevc)，预览可能无法播放。
    # 这里我们假设用户参数是浏览器兼容的，或者用户只关心压缩率/画质。
    # 为了保证能播放，我们强制后缀 .mp4，但编码器使用用户参数。
    # 如果用户选了 hevc，Chrome 可能播放不了（取决于硬件），但这是预览的局限性。
    
    req.params.format = "mp4" # 强制预览为 mp4 容器
    req.params.scodec = "none" # 预览时禁用字幕，防止 TS 图形字幕(dvb_sub)转 MP4 失败
    
    # 截取 10 秒
    input_options = ["-ss", str(start_time), "-t", "10"]
    
    cmd = build_ffmpeg_cmd(input_path, preview_path, req.params, input_options)
    
    try:
        # 同步执行预览生成 (通常 10秒片段很快)
        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if process.returncode != 0:
             raise RuntimeError(f"FFmpeg Preview Error: {process.stdout}")
             
        # 计算预估体积和压缩率
        preview_size = preview_path.stat().st_size
        source_size = input_path.stat().st_size
        
        estimate_info = {
            "source_size": source_size,
            "preview_size": preview_size,
            "duration": duration,
            "estimated_full_size": 0,
            "compression_ratio": 0.0
        }

        if duration > 0:
            # 简单估算：预览10秒 -> 完整时长
            # 注意：如果实际截取不足10秒（视频短于10秒），这里的估算会有偏差，但通常预览针对长视频
            actual_preview_duration = min(10.0, duration)
            ratio = duration / actual_preview_duration
            estimated_full_size = preview_size * ratio
            
            estimate_info["estimated_full_size"] = int(estimated_full_size)
            if source_size > 0:
                # 压缩率：(原大小 - 预估大小) / 原大小
                estimate_info["compression_ratio"] = (source_size - estimated_full_size) / source_size

        # 返回预览 URL 和 统计信息
        preview_url = f"/static/previews/{preview_filename}"
        return {
            "preview_url": preview_url,
            "stats": estimate_info
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 挂载静态文件（确保放在最后，避免覆盖 API 路由）
# 注意：我们需要先创建 static 目录
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
