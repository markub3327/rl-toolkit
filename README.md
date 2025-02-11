# RL Toolkit

[![Release](https://img.shields.io/github/release/markub3327/rl-toolkit)](https://github.com/markub3327/rl-toolkit/releases)
![Tag](https://img.shields.io/github/v/tag/markub3327/rl-toolkit)
[![Issues](https://img.shields.io/github/issues/markub3327/rl-toolkit)](https://github.com/markub3327/rl-toolkit/issues)
![Commits](https://img.shields.io/github/commit-activity/w/markub3327/rl-toolkit)
![Languages](https://img.shields.io/github/languages/count/markub3327/rl-toolkit)
![Size](https://img.shields.io/github/repo-size/markub3327/rl-toolkit)

## Papers
  * [**Playing Flappy Bird Based on Motion Recognition Using a Transformer Model and LIDAR Sensor**](https://www.mdpi.com/1424-8220/24/6/1905)
  * [**Soft Actor-Critic**](https://arxiv.org/abs/1812.05905)
  * [**Generalized State-Dependent Exploration**](https://arxiv.org/abs/2005.05719)
  * [**Reverb: A framework for experience replay**](https://arxiv.org/abs/2102.04736)
  * [**Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics**](https://arxiv.org/abs/2005.04269)
  * [**Acme: A Research Framework for Distributed Reinforcement Learning**](https://arxiv.org/abs/2006.00979)
  * [**Dueling Network Architectures for Deep Reinforcement Learning**](https://arxiv.org/abs/1511.06581)
  * [**Attention Is All You Need**](https://arxiv.org/abs/1706.03762)
  * [**An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**](https://arxiv.org/abs/2010.11929)

## Installation with PyPI

### On PC AMD64 with Ubuntu/Debian

  1. Install dependences
      ```sh
      apt update -y
      apt install swig -y
      ```
  2. Install RL-Toolkit
      ```sh
      pip3 install rl-toolkit[all]
      ```
  3. Run (for **Server**)
      ```sh
      rl_toolkit rl_toolkit -c ./config/sac.yaml -a sac -e BipedalWalkerHardcore-v3 server
      ```
     Run (for **Agent**)
      ```sh
      rl_toolkit -c ./config/sac.yaml -a sac -e BipedalWalkerHardcore-v3 agent
      ```
     Run (for **Learner**)
      ```sh
      rl_toolkit -c ./config/sac.yaml -a sac -e BipedalWalkerHardcore-v3 learner --db_server 192.168.1.2
      ```
     Run (for **Tester**)
      ```sh
      rl_toolkit -c ./config/sac.yaml -a sac -e BipedalWalkerHardcore-v3 tester -f save/model/actor.h5
      ```
  
### On NVIDIA Jetson
 
  1. Install dependences
      <br>Tensorflow for JetPack, follow instructions [here](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html) for installation.
      
      ```sh
      sudo apt install swig -y
      ```
  2. Install Reverb
  <br>Download Bazel 3.7.2 for arm64, [here](https://github.com/bazelbuild/bazel)
      ```sh
      mkdir ~/bin
      mv ~/Downloads/bazel-3.7.2-linux-arm64 ~/bin/bazel
      chmod +x ~/bin/bazel
      export PATH=$PATH:~/bin
      ```
      Clone Reverb with version that corespond with TF verion installed on NVIDIA Jetson !
      ```sh
      git clone https://github.com/deepmind/reverb
      cd reverb/
      git checkout r0.9.0
      ```
      Make changes in Reverb before building !
      <br>In .bazelrc
      ```bazel
      - build:manylinux2010 --crosstool_top=//third_party/toolchains/preconfig/ubuntu16.04/gcc7_manylinux2010:toolchain
      + # build:manylinux2010 --crosstool_top=//third_party/toolchains/preconfig/ubuntu16.04/gcc7_manylinux2010:toolchain
 
      - build --copt=-mavx --copt=-DEIGEN_MAX_ALIGN_BYTES=64
      + build --copt=-DEIGEN_MAX_ALIGN_BYTES=64
      ```
      In WORKSPACE
      ```bazel
      - PROTOC_SHA256 = "15e395b648a1a6dda8fd66868824a396e9d3e89bc2c8648e3b9ab9801bea5d55"
      + # PROTOC_SHA256 = "15e395b648a1a6dda8fd66868824a396e9d3e89bc2c8648e3b9ab9801bea5d55"
      + PROTOC_SHA256 = "7877fee5793c3aafd704e290230de9348d24e8612036f1d784c8863bc790082e"
      ``` 
      In oss_build.sh
      ```bazel
      -  bazel test -c opt --copt=-mavx --config=manylinux2010 --test_output=errors //reverb/cc/...
      +  bazel test -c opt --copt="-march=armv8-a+crypto" --test_output=errors //reverb/cc/...
 
      # Builds Reverb and creates the wheel package.
      -  bazel build -c opt --copt=-mavx $EXTRA_OPT --config=manylinux2010 reverb/pip_package:build_pip_package
      +  bazel build -c opt --copt="-march=armv8-a+crypto" $EXTRA_OPT reverb/pip_package:build_pip_package
      ```
      In reverb/cc/platform/default/repo.bzl
      ```bazel 
      urls = [
         -        "https://github.com/protocolbuffers/protobuf/releases/download/v%s/protoc-%s-linux-x86_64.zip" % (version, version),
         +        "https://github.com/protocolbuffers/protobuf/releases/download/v%s/protoc-%s-linux-aarch_64.zip" % (version, version),
      ]
      ```

     In reverb/pip_package/build_pip_package.sh
     ```sh
     -  "${PYTHON_BIN_PATH}" setup.py bdist_wheel ${PKG_NAME_FLAG} ${RELEASE_FLAG} ${TF_VERSION_FLAG} --plat manylinux2010_x86_64 > /dev/null
     +  "${PYTHON_BIN_PATH}" setup.py bdist_wheel ${PKG_NAME_FLAG} ${RELEASE_FLAG} ${TF_VERSION_FLAG}  > /dev/null
      ```
      Build and install
      ```sh
      bash oss_build.sh --clean true --tf_dep_override "tensorflow~=2.9.1" --release --python "3.8"
      bash ./bazel-bin/reverb/pip_package/build_pip_package --dst /tmp/reverb/dist/ --release
      pip3 install /tmp/reverb/dist/dm_reverb-*
      ```
      Cleaning
      ```sh
      cd ../
      rm -R reverb/
      ```
  3. Install RL-Toolkit
      ```sh
      pip3 install rl-toolkit
      ```

## Environments

  | Environment              | Observation space |  Observation bounds  | Action space |   Action bounds    | Reward bounds |
  | ------------------------ |:-----------------:|:--------------------:|:------------:|:------------------:| :-----------: |
  | BipedalWalkerHardcore-v3 |      (24, )       |     [-inf, inf]      |    (4, )     |    [-1.0, 1.0]     | [-1.0, 1.0] |
  | FlappyBird-v0 |     (16, 180)     | [0, d<sub>max</sub>] |    (2, )     | {DO NOTHING, FLAP} | [-1.0, 1.0] |
  
## Results

  | Environment              |                                         SAC<br> + gSDE                                         | SAC<br> + gSDE<br>+ Huber loss |                                    SAC<br> + TQC<br> + gSDE                                     | Q-Learning | RL-Toolkit |
  | ------------------------ |:----------------------------------------------------------------------------------------------:|:------------------------------:|:-----------------------------------------------------------------------------------------------:|:----------:|:----------:|
  | BipedalWalkerHardcore-v3 | 13 ± 18[<sup>(1)</sup>](https://sb3-contrib.readthedocs.io/en/stable/modules/tqc.html#results) |         **239 ± 118**          | 228 ± 18[<sup>(1)</sup>](https://sb3-contrib.readthedocs.io/en/stable/modules/tqc.html#results) |     -      | 205 ± 134  |
  | FlappyBird-v0 |                                               -                                                |-|                                                -                                                |  209.298[<sup>(2)</sup>](https://arxiv.org/pdf/2003.09579)   |   13 156   |
  
![dm_ant_ball_sac](https://raw.githubusercontent.com/markub3327/rl-toolkit/master/img/dm_ant_ball_sac.gif)

## Releases

   * SAC + gSDE + Huber loss<br> &emsp; is stored here, [branch r2.0](https://github.com/markub3327/rl-toolkit/tree/r2.0)
   * SAC + TQC + gSDE + LogCosh + Reverb<br> &emsp; is stored here, [branch r4.0](https://github.com/markub3327/rl-toolkit/tree/r4.1)
   * DQN + SAC agents [branch r5.0](https://github.com/markub3327/rl-toolkit/tree/r5.0)

----------------------------------

**Frameworks:** Tensorflow, DeepMind Reverb, Gymnasium, DeepMind Control Suite, WanDB, OpenCV
