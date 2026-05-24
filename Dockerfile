ARG CUDA_VERSION=12.8.0
ARG UBUNTU_VERSION=22.04
ARG TORCH_VERSION=2.7.1
ARG TORCH_CUDA_TAG=cu128

FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu${UBUNTU_VERSION} AS builder

ARG TORCH_VERSION
ARG TORCH_CUDA_TAG

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-venv \
        python3.10-dev \
        python3-pip \
        build-essential \
        git \
        ca-certificates \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --upgrade pip

RUN pip install --index-url https://download.pytorch.org/whl/${TORCH_CUDA_TAG} \
        torch==${TORCH_VERSION} torchaudio==${TORCH_VERSION}

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

RUN pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.6.3+cu128torch2.7-cp310-cp310-linux_x86_64.whl

RUN find /opt/venv -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true

FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/opt/app

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        gcc \
        libsndfile1 \
        ca-certificates \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

COPY app /opt/app
COPY --from=builder /opt/venv /opt/venv

WORKDIR /opt/app

CMD ["python", "run_gradio.py", "--model", "medium"]

