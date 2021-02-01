# RL training toolkit

[![Release](https://img.shields.io/github/release/markub3327/rl-baselines)](https://github.com/markub3327/rl-baselines/releases)
![Tag](https://img.shields.io/github/v/tag/markub3327/rl-baselines)

[![Issues](https://img.shields.io/github/issues/markub3327/rl-baselines)](https://github.com/markub3327/rl-baselines/issues)
![Commits](https://img.shields.io/github/commit-activity/w/markub3327/rl-baselines)

![Languages](https://img.shields.io/github/languages/count/markub3327/rl-baselines)
![Size](https://img.shields.io/github/repo-size/markub3327/rl-baselines)

## Papers

  * **Soft Actor-Critic** (https://arxiv.org/pdf/1812.05905.pdf)
  * **Twin Delayed DDPG** (https://arxiv.org/pdf/1802.09477.pdf)
  * **Generalized State-Dependent Exploration** (https://arxiv.org/pdf/2005.05719.pdf)

## Setting up container

#### YOU MUST HAVE INSTALLED DOCKER !!!

```shell
# 1. Build the Docker image
./build.sh

# 2. Run 'RL toolkit' in the container
./run.sh python3 main.py [-h]
```

## Using

```shell
# Run RL toolkit for training
python3 main.py [-h] -env ENV_NAME -s PATH_TO_MODEL [--wandb]

# Run RL toolkit for testing
python3 main.py [-h] -env ENV_NAME --model_a PATH_TO_MODEL --test [--wandb]
```

## Topology

<p align="center"><b>Actor (Twin Delayed DDPG)</b></p>
<p align="center">
  <img src="img/model_A_TD3.png" alt="actor">
</p>

<p align="center"><b>Critic (Twin Delayed DDPG)</b></p>
<p align="center">
  <img src="img/model_C_TD3.png" alt="critic">
</p>

<p align="center"><b>Actor (Soft Actor-Critic)</b></p>
<p align="center">
  <img src="img/model_A_SAC.png" alt="actor">
</p>

<p align="center"><b>Critic (Soft Actor-Critic)</b></p>
<p align="center">
  <img src="img/model_C_SAC.png" alt="critic">
</p>

## Tested environments
  
  * MountainCarContinuous-v0
  * BipedalWalker-v3
  * BipedalWalkerHardcore-v3
  * LunarLanderContinuous-v2
  * Walker2DBulletEnv-v0
  * AntBulletEnv-v0

<p align="center"><b>Summary</b></p>
<p align="center">
  <img src="img/results.png" alt="results">
</p>
<p align="center"><a href="https://app.wandb.ai/markub/rl-baselines" target="_blank">For more charts click here.</a></p>

**Framework:** Tensorflow 2.3.1
</br>
**Languages:** Python 3.8 
</br>
**Author**: Martin Kubovcik
