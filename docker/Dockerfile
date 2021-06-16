# nainstaluj Ubuntu 20.04 LTS
FROM ubuntu:20.04

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
    graphviz \
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
RUN python3 -m pip install --no-cache-dir rl-toolkit

###########################################
# Entry point
###########################################
COPY docker/entrypoint.sh /root/
RUN chmod +x /root/entrypoint.sh
ENTRYPOINT ["/root/entrypoint.sh"]
