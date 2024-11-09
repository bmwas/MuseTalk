# Use an NVIDIA CUDA base image with Ubuntu 22.04 and CUDA 11.7
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip git wget unzip ffmpeg git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install OpenMIM
RUN pip install --no-cache-dir -U openmim

# Install PyTorch compatible with CUDA 11.7
RUN pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --index-url https://download.pytorch.org/whl/cu117

# Install MMEngine and other MM packages
RUN mim install mmengine
RUN mim install "mmcv>=2.0.1"
RUN mim install "mmdet>=3.1.0"
RUN mim install "mmpose>=1.1.0"

# Install git-lfs
RUN git lfs install

# Clone the MuseTalk repository into /app
RUN git clone https://github.com/TMElyralab/MuseTalk.git /app

# Set the working directory
WORKDIR /app

# Install Python requirements
RUN pip install -r requirements.txt

# Download models.zip from Hugging Face
RUN git clone https://huggingface.co/Benson/musetalkmodels /tmp/musetalkmodels

# Unzip models.zip into /app/models
RUN unzip /tmp/musetalkmodels/models.zip -d /app

# Clean up temporary directories
RUN rm -rf /tmp/musetalkmodels

# Set environment variable for FFMPEG_PATH
ENV FFMPEG_PATH=/app/ffmpeg

# Download ffmpeg-static
RUN wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz -O ffmpeg.tar.xz \
    && tar -xf ffmpeg.tar.xz \
    && mv ffmpeg-*-amd64-static $FFMPEG_PATH \
    && rm ffmpeg.tar.xz

# Expose port if needed (optional)
EXPOSE 8000

# Set the default command to bash
CMD ["bash"]

