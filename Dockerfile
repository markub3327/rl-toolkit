# nainstaluj Ubuntu 20.04 LTS
FROM ubuntu:20.04

# nastav jazyk
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# nastav apt-get
ARG DEBIAN_FRONTEND=noninteractive

# nainstaluj python3 a vycisti balicky
RUN apt-get update && apt-get install -y \
    curl \
    python3 \
    python3-pip \
    python3-numpy \
    git \
    && rm -rf /var/lib/apt/lists/*

# Some TF tools expect a "python" binary
RUN ln -s $(which python3) /usr/local/bin/python

###########################################
# OpenAI Gym
# Source: https://github.com/openai/gym
###########################################
RUN apt-get -y update && apt-get install -y \
        unzip \
        libglu1-mesa-dev \
        libgl1-mesa-dev \
        libosmesa6-dev \
        xvfb \
        patchelf \
        ffmpeg \
        cmake \
        swig \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/local/
RUN git clone https://github.com/openai/gym.git
WORKDIR /usr/local/gym/
RUN python3 -m pip install --no-cache-dir -e '.[box2d,classic_control]'

###########################################
# Tensorflow
# Source: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/dockerfiles/dockerfiles/cpu.Dockerfile
###########################################
RUN python3 -m pip --no-cache-dir install --upgrade \
    "pip" \
    setuptools
# Options:
#   tensorflow
#   tensorflow-gpu
#   tf-nightly
#   tf-nightly-gpu
# Set --build-arg TF_PACKAGE_VERSION=1.11.0rc0 to install a specific version.
# Installs the latest version by default.
ARG TF_PACKAGE=tensorflow
ARG TF_PACKAGE_VERSION=
RUN python3 -m pip install --no-cache-dir ${TF_PACKAGE}${TF_PACKAGE_VERSION:+==${TF_PACKAGE_VERSION}}

###########################################
# Dependencies
###########################################
RUN python3 -m pip --no-cache-dir install -r requirements.txt

# nastav pracovny priecinok na /root
WORKDIR /root
