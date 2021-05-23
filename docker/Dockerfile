# nainstaluj Ubuntu 20.04 LTS
FROM ubuntu:20.04

# Maintainer
LABEL org.opencontainers.image.source = "https://github.com/markub3327/rl-toolkit"
LABEL org.opencontainers.image.authors="markub3327@gmail.com"

# nastav jazyk
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# nastav apt-get
ARG DEBIAN_FRONTEND=noninteractive

# nainstaluj python3 a vycisti balicky
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    python3-numpy \
    && rm -rf /var/lib/apt/lists/*

###########################################
# Dependencies
###########################################
RUN apt-get -y update && apt-get install -y \
        libglu1-mesa-dev \
        libgl1-mesa-dev \
        libosmesa6-dev \
        xvfb \
        patchelf \
        ffmpeg \
        swig \
        && rm -rf /var/lib/apt/lists/*

###########################################
# RL Toolkit
###########################################
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

RUN mkdir /root/rl-toolkit
COPY . /root/rl-toolkit/
WORKDIR /root/rl-toolkit

RUN python3 -m pip install --no-cache-dir -r requirements.txt

###########################################
# Entry point
###########################################
COPY docker/entrypoint.sh /root/
RUN chmod +x /root/entrypoint.sh
ENTRYPOINT ["/root/entrypoint.sh"]
