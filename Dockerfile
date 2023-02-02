# Author Cameron-git
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive 

WORKDIR /app

RUN apt-get update
RUN apt-get upgrade -y

RUN apt-get install -y git python3-pip
RUN apt-get install -y gcc-10 g++-10

# Pytorch
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# CUDA 11.6
RUN apt-get install -y wget zlib1g
RUN apt-key del 7fa2af80
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb
RUN apt-get update
RUN apt-get install -y cuda-toolkit-11-6
ENV PATH="/usr/local/cuda/bin:$PATH"

# CUDNN 8.7
COPY "./cudnn-local-repo-ubuntu2004-8.7.0.84_1.0-1_amd64.deb" "./cudnn.deb"
RUN dpkg -i ./cudnn.deb
RUN cp /var/cudnn-local-repo-ubuntu2004-8.7.0.84/cudnn-local-A3837CDF-keyring.gpg /usr/share/keyrings/
RUN apt-get update
RUN apt-get install -y libcudnn8=8.7.0.84-1+cuda11.*
RUN apt-get install -y libcudnn8-dev=8.7.0.84-1+cuda11.*
RUN apt-get install -y libcudnn8-samples=8.7.0.84-1+cuda11.*