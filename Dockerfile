FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# 필수 패키지 설치 + deadsnakes PPA로 Python 3.7 설치
RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    wget \
    curl \
    git \
    liblzma-dev \
    libssl-dev \
    libmpich-dev \
    libopenmpi-dev \
    libbz2-dev \
    libreadline-dev \
    libffi-dev \
    libsqlite3-dev \
    zlib1g-dev \
    libncurses5-dev \
    libnss3-dev \
    cmake \
    ninja-build && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y python3.7-venv \
    apt-get install gstreamer1.0-plugins-base gstreamer1.0-plugins-ugly \   
    apt-get install -y libsndfile1-dev \
    apt-get install -y libgl1-mesa-glx \
    apt-get install -y python3.7 python3.7-dev python3.7-distutils && \
    curl -sS https://bootstrap.pypa.io/pip/3.7/get-pip.py | python3.7

# 심볼릭 링크 설정
RUN ln -sf /usr/bin/python3.7 /usr/bin/python && \
    ln -sf /usr/local/bin/pip3.7 /usr/bin/pip && \
    ln -sf /usr/local/bin/pip3.7 /usr/bin/pip3

# PyTorch 설치 (CUDA 11.6 호환)
RUN pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 \
    -f https://download.pytorch.org/whl/torch_stable.html

# OpenAI jukebox 설치
RUN pip install git+https://github.com/openai/jukebox.git

# 작업 디렉토리 설정
WORKDIR /workspace

CMD ["/bin/bash"]
