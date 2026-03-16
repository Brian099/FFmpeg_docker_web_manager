FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 安装工具并在线下载 FFmpeg (多架构)
ARG TARGETARCH
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    xz-utils \
    procps \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    case "${TARGETARCH}" in \
        amd64) pkg="ffmpeg-master-latest-linux64-gpl.tar.xz" ;; \
        arm64) pkg="ffmpeg-master-latest-linuxarm64-gpl.tar.xz" ;; \
        *) echo "Unsupported arch: ${TARGETARCH}"; exit 1 ;; \
    esac; \
    url="https://github.com/BtbN/FFmpeg-Builds/releases/latest/download/${pkg}"; \
    mkdir -p /opt/ffmpeg; \
    curl -L --retry 3 -o /tmp/ffmpeg.tar.xz "${url}"; \
    tar -xJf /tmp/ffmpeg.tar.xz -C /opt/ffmpeg --strip-components=1; \
    rm -f /tmp/ffmpeg.tar.xz; \
    ln -sf /opt/ffmpeg/bin/ffmpeg /usr/local/bin/ffmpeg; \
    ln -sf /opt/ffmpeg/bin/ffprobe /usr/local/bin/ffprobe; \
    ffmpeg -version; \
    ffprobe -version

# 复制依赖并安装
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 复制应用代码
COPY . /app

# 创建数据目录和配置目录
ENV DATA_DIR=/data
ENV CONFIG_DIR=/config
RUN mkdir -p /data/input /data/output /data/todo /config

EXPOSE 8000

# 启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
