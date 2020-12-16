FROM ubuntu:18.04 as base

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y \
    curl \
    python3 \
    python3-pip \
    python3-numpy \
    python3-dev \
    git

RUN python3 -m pip --no-cache-dir install --upgrade \
    "pip<20.3" \
    setuptools \
    wandb \
    tensorflow_probability \
    pybullet

# Some TF tools expect a "python" binary
RUN ln -s $(which python3) /usr/local/bin/python

###########################################
# X11 VNC XVFB
# see: https://github.com/fcwu/docker-ubuntu-vnc-desktop
###########################################
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        software-properties-common \
        wget \
        supervisor \
        vim-tiny \
        net-tools \ 
        xz-utils \
        dbus-x11 x11-utils alsa-utils \
        mesa-utils libgl1-mesa-dri \
        lxde x11vnc xvfb \
    && apt-get autoclean \
    && apt-get autoremove \
    && rm -rf /var/lib/apt/lists/*

# tini for subreap                                   
ARG TINI_VERSION=v0.9.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /bin/tini
RUN chmod +x /bin/tini

###########################################
# OpenAI Gym
# see: https://github.com/openai/gym
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
        swig\
    && rm -rf /var/lib/apt/lists/*

# install gym
RUN cd /usr/local/ \
    && git clone https://github.com/openai/gym.git \
    && cd /usr/local/gym/ \
    && python3 -m pip install --no-cache-dir -e '.[box2d,classic_control]'

###########################################
# Tensorflow
# see: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/dockerfiles/dockerfiles/cpu.Dockerfile
###########################################
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

# vnc port
EXPOSE 5900

#COPY ./docker_image/docker_entrypoint.sh /root/
#COPY ./docker_image/supervisord.conf /etc/supervisor/supervisord.conf
WORKDIR /root
# run all services
#ENTRYPOINT [ "/root/docker_entrypoint.sh" ]
