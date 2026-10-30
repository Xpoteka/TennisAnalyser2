# Phase 0 spike image: Intel Arc (xe) GPU userspace + PyTorch XPU + OpenVINO.
# The host only provides the kernel driver and /dev/dri; everything else lives here.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# Intel compute runtime (OpenCL for OpenVINO, Level Zero for PyTorch XPU).
RUN apt-get update \
 && apt-get install -y --no-install-recommends software-properties-common gpg-agent ca-certificates \
 && add-apt-repository -y ppa:kobuk-team/intel-graphics \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      libze-intel-gpu1 libze1 intel-opencl-icd clinfo pciutils \
      python3 python3-venv \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# Unpinned until the first green run on the B70; env_check.json records the exact versions.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/xpu
RUN pip install torch torchvision --index-url ${TORCH_INDEX_URL}

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

WORKDIR /work
COPY env_check.py /work/

CMD ["python", "env_check.py", "--out", "/work/out/env_check.json"]
