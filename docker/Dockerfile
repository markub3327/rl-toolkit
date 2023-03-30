# nainstaluj Ubuntu 20.04 LTS
FROM ubuntu:20.04

# nastav jazyk
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# nastav apt-get
ARG DEBIAN_FRONTEND=noninteractive

###########################################
# Dependencies
###########################################
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    python3 \
    python3-dev \
    graphviz \
    libglu1-mesa-dev \
    libgl1-mesa-dev \
    libosmesa6-dev \
    xvfb \
    patchelf \
    ffmpeg \
    swig \
    && rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py
RUN python3 get-pip.py

###########################################
# RL Toolkit
###########################################
WORKDIR /root/
RUN git clone https://github.com/markub3327/rl-toolkit.git
WORKDIR /root/rl-toolkit

RUN python3 -mpip install --no-cache-dir -e .[all]

###########################################
# Entry point
###########################################
COPY docker/entrypoint.sh /root/
RUN chmod +x /root/entrypoint.sh
ENTRYPOINT ["/root/entrypoint.sh"]

CMD [ "/bin/bash" ]