# RL toolkit

[![Release](https://img.shields.io/github/release/markub3327/rl-toolkit)](https://github.com/markub3327/rl-toolkit/releases)
![Tag](https://img.shields.io/github/v/tag/markub3327/rl-toolkit)

[![Issues](https://img.shields.io/github/issues/markub3327/rl-toolkit)](https://github.com/markub3327/rl-toolkit/issues)
![Commits](https://img.shields.io/github/commit-activity/w/markub3327/rl-toolkit)

![Languages](https://img.shields.io/github/languages/count/markub3327/rl-toolkit)
![Size](https://img.shields.io/github/repo-size/markub3327/rl-toolkit)

## Papers
  * [**Soft Actor-Critic**](https://arxiv.org/pdf/1812.05905.pdf)
  * [**Twin Delayed DDPG**](https://arxiv.org/pdf/1802.09477.pdf)
  * [**Generalized State-Dependent Exploration**](https://arxiv.org/pdf/2005.05719.pdf)

## Setting up container

##### YOU MUST HAVE INSTALLED DOCKER !!!

### Build the Docker image

```shell
docker build -t markub/rl-toolkit .
```

### Run the container

```shell
docker run -it --rm -v $PWD:/root/rl-toolkit markub/rl-toolkit:latest python3 training.py
```

## Run

```shell
# Run training
python3 training.py [-h] -alg sac -env ENV_NAME -s PATH_TO_MODEL_FOLDER [--wandb]

# Run testing
python3 testing.py [-h] -alg td3 -env ENV_NAME -f PATH_TO_MODEL_FOLDER [--wandb]
```

## Tested environments

  | Environment              | Observation space | Observation bounds | Action space | Action bounds |
  | ------------------------ | :---------------: | :----------------: | :----------: | :-----------: |
  | MountainCarContinuous-v0 | (2, ) | [-1.2 , 0.6] | (1, ) | [-1.0 , 1.0] |
  | BipedalWalker-v3         | (24, ) | [-inf , inf] | (4, ) | [-1.0 , 1.0] | 
  | BipedalWalkerHardcore-v3 | (24, ) | [-inf , inf] | (4, ) | [-1.0 , 1.0] |
  | LunarLanderContinuous-v2 | (8, ) | [-inf , inf] | (2, ) | [-1.0 , 1.0] |
  | Walker2DBulletEnv-v0     | (22, ) | [-inf , inf] | (6, ) | [-1.0 , 1.0] |
  | AntBulletEnv-v0          | (28, ) | [-inf , inf] | (8, ) | [-1.0 , 1.0] |
  | HalfCheetahBulletEnv-v0  | (26, ) | [-inf , inf] | (6, ) | [-1.0 , 1.0] |
  | HopperBulletEnv-v0       | (15, ) | [-inf , inf] | (3, ) | [-1.0 , 1.0] |


## Results

<p align="center"><b>Summary</b></p>
<p align="center">
  <a href="https://wandb.ai/markub/rl-toolkit?workspace=user-markub" target="_blank"><img src="img/results.png" alt="results"></a>
</p>

<p align="center"><b>Return from game</b></p>

  | Environment              | gSDE<br>[(source)](https://paperswithcode.com/paper/generalized-state-dependent-exploration-for) | gSDE<br>+ Huber loss |
  | ------------------------ | :---: | :-----------------: |
  | MountainCarContinuous-v0 | ± | - |
  | BipedalWalker-v3         | ± | - |
  | BipedalWalkerHardcore-v3 | ± | - |
  | LunarLanderContinuous-v2 | ± | - |
  | Walker2DBulletEnv-v0     | 2270 ± 28 | **2732 ± 96** |
  | AntBulletEnv-v0          | 3106 ± 61 | **3458 ± 120** |
  | HalfCheetahBulletEnv-v0  | 2945 ± 95 | 2915 ± 230 |
  | HopperBulletEnv-v0       | 2515 ± 50 | - |

----------------------------------

**Framework:** Tensorflow 2.4
<br>
**Languages:** Python 3.8
<br>
**Author**: Martin Kubovcik
