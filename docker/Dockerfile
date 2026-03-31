# Base image with CUDA support
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/conda/bin:$PATH"

# Install system dependencies (Required for MuJoCo & Graphics)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    git \
    build-essential \
    cmake \
    patchelf \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libglew-dev \
    libosmesa6-dev \
    libglfw3 \
    libglfw3-dev \
    unzip \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -afy

WORKDIR /workspace

# Copy the exact Conda environment file from the DP submodule
COPY third_party/diffusion_policy/conda_environment.yaml /tmp/environment.yaml

# Accept Conda Terms of Service for default channels
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# CRITICAL FIX: Remove problematic pip dependencies from the yaml to avoid source compilation errors
# We remove 'dm-control', 'mujoco', 'free-mujoco-py' and 'robosuite' from the yaml
RUN sed -i '/dm-control/d' /tmp/environment.yaml && \
    sed -i '/mujoco==/d' /tmp/environment.yaml && \
    sed -i '/free-mujoco-py/d' /tmp/environment.yaml && \
    sed -i '/robosuite/d' /tmp/environment.yaml

# Create the Conda environment (this will now pass without the broken pip builds)
RUN conda env create -f /tmp/environment.yaml && conda clean -afy

# Set the new Conda environment as the default PATH
ENV PATH="/opt/conda/envs/robodiff/bin:$PATH"

# Force install the pre-compiled binary versions of MuJoCo and other simulation libs
# We use --prefer-binary to ensure pip downloads wheels instead of compiling from source
RUN pip install --no-cache-dir --prefer-binary \
    mujoco==2.3.3 \
    free-mujoco-py==2.1.6 \
    dm-control==1.0.11

# Install robosuite directly from the specified GitHub commit
RUN pip install --no-cache-dir "git+https://github.com/cheng-chi/robosuite.git@277ab9588ad7a4f4b55cf75508b44aa67ec171f0"

# Install custom kinematics requirements for KC-DP
RUN pip install --no-cache-dir \
    pinocchio \
    spatialmath-python

# Keep the container running
CMD ["tail", "-f", "/dev/null"]
