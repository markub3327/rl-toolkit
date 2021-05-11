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
# Dependencies
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
# RL Toolkit
# Source: https://github.com/markub3327/rl-toolkit
###########################################
WORKDIR /root/
RUN git clone https://github.com/markub3327/rl-toolkit
WORKDIR /root/rl-toolkit

COPY requirements.txt /tmp/
RUN python3 -m pip --no-cache-dir install -r /tmp/requirements.txt

###########################################
# Entry point
###########################################
COPY docker/docker_entrypoint.sh /root
RUN chmod +x /root/docker_entrypoint.sh
ENTRYPOINT ["/root/docker_entrypoint.sh"]
