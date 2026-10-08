# ProCIR training image: dependencies only. The repository (code + data/ + runs/) is bind-mounted
# at /workspace by docker/compose.yaml, so code changes need no rebuild.
# "devel" base: causal-conv1d compiles CUDA code at install time and needs nvcc.
ARG BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/workspace/.cache/huggingface \
    PYTHONPATH=/workspace/src:/workspace/external/FashionMV

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY requirements-train.txt /tmp/requirements-train.txt
RUN pip install -r /tmp/requirements-train.txt

# Qwen3.5 linear-attention kernels. Installed separately: a failed causal-conv1d build must not
# take flash-linear-attention down with it (that is what happened on Kaggle).
# Ampere (8.0, 8.6), Ada (8.9) and Hopper (9.0); narrow the list to build faster.
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
ARG MAX_JOBS=8
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} MAX_JOBS=${MAX_JOBS}
RUN pip install flash-linear-attention
RUN pip install causal-conv1d --no-build-isolation

WORKDIR /workspace
CMD ["python", "scripts/check_env.py"]
