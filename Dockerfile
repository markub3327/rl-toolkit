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

###########################################
# Tensorflow
# Source: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/dockerfiles/dockerfiles/cpu.Dockerfile
###########################################
RUN python3 -m pip --no-cache-dir install --upgrade \
    "pip" \
    setuptools

###########################################
# Dependencies
###########################################
COPY requirements.txt /tmp/
RUN python3 -m pip --no-cache-dir install -r /tmp/requirements.txt

# vytvor pracovny priecinok pre RL nastroje
RUN mkdir /root/rl-toolkit
WORKDIR /root/rl-toolkit

# nastav vychodiskovy bod pre kontajner
COPY docker_entrypoint.sh /root
RUN chmod +x /root/docker_entrypoint.sh
ENTRYPOINT ["/root/docker_entrypoint.sh"]
