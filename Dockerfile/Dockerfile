# Docker file for PyTorch 

FROM nvidia/cuda:11.2.0-devel-ubuntu20.04

#
# USER root
#

USER root

ARG UID=1058
ARG GID=1060
ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y -qq --no-install-recommends \
        tmux sudo nano htop wget less \
        iputils-ping net-tools \
        git curl \
        python3-dev python3-pip python3-tk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
RUN apt install --yes build-essential libssl-dev cmake
RUN apt install --yes git git-lfs
RUN git lfs install
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
