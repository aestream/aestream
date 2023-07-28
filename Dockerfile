# Author Cameron-git
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive 

WORKDIR /app

# Install dependencies
RUN apt-get update
RUN apt-get install -y git python3-pip --no-install-recommends
RUN apt-get install -y gcc-10 g++-10 --no-install-recommends

# Pytorch
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# CUDA 11.6
RUN apt-get install -y wget zlib1g
RUN apt-key del 7fa2af80
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb
RUN apt-get update
RUN apt-get install -y cuda-toolkit-12-0
ENV PATH="/usr/local/cuda/bin:$PATH"

# AEStream
RUN CC=gcc-10 CXX=g++-10 pip install git+https://github.com/aestream/aestream.git