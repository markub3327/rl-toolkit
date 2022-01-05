# Changes

## v4.0.0 (XXX XX, 2022)
### Features 🔊
- Render environments to WanDB
- Grouping of runs in WanDB
- SampleToInsertRatio rate limiter
- Global Gradient Clipping to avoid exploding gradients
- Softplus for numerical stability
- YAML configuration file
- LogCosh instead of Huber loss
- Critic network with Add layer applied on state & action branches
- Custom uniform initializer
- [LayerNormalization + TanH] block for Input layer
- XLA (Accelerated Linear Algebra) compiler
- Optimized Replay Buffer (https://github.com/deepmind/reverb/issues/90)
- split into **Agent**, **Learner**, **Tester** and **Server**
### Bug fixes 🛠️
- Fixed creating of saving path for models
- Fixed model's `summary()`

## v3.2.4 (July 7, 2021)
### Features 🔊
- Reverb
- `setup.py` (package is available on PyPI)
- split into **Agent**, **Learner** and **Tester**
- Use custom model and layer for defining Actor-Critic
- MultiCritic - concatenating multiple critic networks into one network
- Truncated Quantile Critics

## v2.0.2 (May 23, 2021)
### Features 🔊
- update Dockerfile
- update `README.md`
- formatted code by Black & Flake8

## v2.0.1 (April 27, 2021)
### Bug fixes 🛠️
- fixed Critic model

## v2.0.0 (April 22, 2021)
### Features 🔊
- Add Huber loss
- In test mode, rendering to the video file
- Normalized observation by Min-max method
- Remove TD3 algorithm
