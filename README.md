# rl-baselines

[![Release](https://img.shields.io/github/release/markub3327/stable-baselines)](https://github.com/markub3327/stable-baselines/releases)
![Tag](https://img.shields.io/github/v/tag/markub3327/stable-baselines)

[![Issues](https://img.shields.io/github/issues/markub3327/stable-baselines)](https://github.com/markub3327/stable-baselines/issues)
![Commits](https://img.shields.io/github/commit-activity/w/markub3327/stable-baselines)

![Languages](https://img.shields.io/github/languages/count/markub3327/stable-baselines)
![Size](https://img.shields.io/github/repo-size/markub3327/stable-baselines)

## Agents

  * **Soft Actor-Critic** (https://arxiv.org/pdf/1812.05905.pdf)
  * **Twin Delayed DDPG** (https://arxiv.org/pdf/1802.09477.pdf)

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
  * LunarLanderContinuous-v2
  * Walker2DPyBulletEnv-v0

<p align="center"><b>Summary</b></p>
<p align="center">
  <img src="img/results.png" alt="results">
</p>
<p align="center"><a href="https://app.wandb.ai/markub/stable-baselines" target="_blank">For more charts click here.</a></p>

**Framework:** Tensorflow 2.4.0
</br>
**Languages:** Python 3.8 
</br>
**Author**: Martin Kubovcik
