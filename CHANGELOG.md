# Changes

## v3.2.4 (June XX, 2021)
### Features 🔊
- Reverb (+multi-node learning) ✔️
- `setup.py` (package is available on PyPI) ✔️
- split into **agent**, **learner**, **tester** and **server** roles ✔️
- Use custom model and layer for defining Actor-Critic ✔️
- MultiCritic - concatenating multiple critic networks into one network ✔️
- Truncated Quantile Critics + Fully Parameterized Quantile
- Normalization layer (CrossRenorm)
- Without target networks ✔️
- Generating `png` image of the model ✔️

## v2.0.2 (May 23, 2021)
### Bug fixes 🛠️
- update Dockerfile
- update `README.md`
- formatted code by Black & Flake8

## v2.0.1 (April 27, 2021)
### Bug fixes 🛠️
- fix Critic model

## v2.0.0 (April 22, 2021)
### Features 🔊
- Add Huber loss
- In test mode, rendering to the video file
- Normalized observation by Min-max method
- Remove TD3 algorithm
