from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

long_description += "\n\n"

with open("CHANGELOG.md", "r", encoding="utf-8") as fh:
    long_description += fh.read()

setup(
    name="rl-toolkit",
    version="3.2.4",
    description="The RL-Toolkit: A toolkit for developing and comparing your reinforcement learning agents in various games (OpenAI Gym or Pybullet).",  # noqa
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/markub3327/rl-toolkit",
    project_urls={
        "Bug Tracker": "https://github.com/markub3327/rl-toolkit/issues",
    },
    author="Martin Kubovčík",
    author_email="markub3327@gmail.com",
    license="mit",
    packages=[
        package for package in find_packages() if package.startswith("rl_toolkit")
    ],
    keywords=[
        "reinforcement-learning",
        "ml",
        "openai-gym",
        "pybullet",
        "reverb",
        "docker",
        "rl-agents",
        "rl",
        "sac",
        "rl-algorithms",
        "soft-actor-critic",
        "gsde",
        "rl-toolkit",
        "games",
        "tensorflow",
        "wandb",
    ],
    install_requires=[
        "gym",
        "box2d",
        "pybullet",
        "tensorflow",
        "tensorflow_probability",
        "opencv-python",
        "wandb",
        "dm-reverb",
        "pydot",
    ],
    python_requires=">=3.6",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Environment :: Console",
        "Environment :: GPU :: NVIDIA CUDA",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
