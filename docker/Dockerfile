FROM tensorflow/tensorflow:2.14.0-gpu

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
    graphviz \
    libglu1-mesa-dev \
    libgl1-mesa-dev \
    libosmesa6-dev \
    xvfb \
    patchelf \
    ffmpeg \
    swig \
    screen

RUN apt-get clean && rm -rf /tmp/* /var/tmp/*

RUN python -m pip install --no-cache-dir \
                           matplotlib \
                           wandb \
                           pydot \
                           scikit-learn \
                           einops \
                           black[jupyter] \
                           seaborn \
                           opencv-python \
                           isort \
                           flake8 \
                           jupyterlab \
                           jupyterlab-git \
                           keras_cv \
                           tf2onnx \

###########################################
# RL Toolkit
###########################################
WORKDIR /root/
RUN git clone https://github.com/markub3327/rl-toolkit.git
WORKDIR /root/rl-toolkit

RUN python -m pip install --no-cache-dir -e .[all]

###########################################
# Entry point
###########################################
# COPY docker/entrypoint.sh /root/
# RUN chmod +x /root/entrypoint.sh
# ENTRYPOINT ["/root/entrypoint.sh"]

# CMD [ "/bin/bash" ]

###########################################
# Jupyter Lab
###########################################
# Port
EXPOSE 8888/tcp

# Run JupyterLab
CMD ["bash", "-c", "jupyter lab --notebook-dir=/tf --ip 0.0.0.0 --no-browser --allow-root"]