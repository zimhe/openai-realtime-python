FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG UBUNTU_MIRROR=http://mirrors.tuna.tsinghua.edu.cn/ubuntu/
ARG UBUNTU_SECURITY_MIRROR=http://mirrors.tuna.tsinghua.edu.cn/ubuntu/

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN set -eux; \
	sed -i "s|http://archive.ubuntu.com/ubuntu/|${UBUNTU_MIRROR}|g" /etc/apt/sources.list; \
	sed -i "s|http://security.ubuntu.com/ubuntu/|${UBUNTU_SECURITY_MIRROR}|g" /etc/apt/sources.list

# Install system dependencies, including gcc and other libraries
RUN set -eux; \
    apt-get update -o Acquire::Retries=5; \
    apt-get install -y --no-install-recommends \
        -o Acquire::Retries=5 \
        -o Acquire::http::Timeout=30 \
        -o Acquire::https::Timeout=30 \
        software-properties-common gnupg ffmpeg gcc libc++-dev libc++abi-dev portaudio19-dev \
    || (apt-get update -o Acquire::Retries=5 && \
        apt-get install -y --fix-missing --no-install-recommends \
            -o Acquire::Retries=5 \
            -o Acquire::http::Timeout=30 \
            -o Acquire::https::Timeout=30 \
            software-properties-common gnupg ffmpeg gcc libc++-dev libc++abi-dev portaudio19-dev); \
    rm -rf /var/lib/apt/lists/*

# Install Python 3.12 and pip
RUN set -eux; \
    add-apt-repository ppa:deadsnakes/ppa; \
    apt-get update -o Acquire::Retries=5; \
    apt-get install -y --no-install-recommends \
        -o Acquire::Retries=5 \
        -o Acquire::http::Timeout=30 \
        -o Acquire::https::Timeout=30 \
        python3.12 python3-pip python3.12-dev python3.12-venv \
    || (apt-get update -o Acquire::Retries=5 && \
        apt-get install -y --fix-missing --no-install-recommends \
            -o Acquire::Retries=5 \
            -o Acquire::http::Timeout=30 \
            -o Acquire::https::Timeout=30 \
            python3.12 python3-pip python3.12-dev python3.12-venv); \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 100; \
    python3 -m ensurepip --upgrade; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY realtime_agent realtime_agent

EXPOSE 8080

# Default command to run the app
CMD ["python3", "-m", "realtime_agent.main", "server"]
