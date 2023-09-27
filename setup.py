from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

long_description += "\n\n"

with open("README.md", "r", encoding="utf-8") as fh:
    long_description += fh.read()

extras = {
    "all": ["tensorflow", "dm-reverb", "flappy-bird-gymnasium"],
    "reverb": ["dm-reverb"],
    "tf": ["tensorflow"],
}

setup(
    name="rl-toolkit",
    version="5.0.0",
    author="Martin Kubovcik",
    author_email="markub3327@gmail.com",
    description="RL-Toolkit: A Research Framework for Robotics",  # noqa
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/markub3327/rl-toolkit",
    project_urls={
        "Bug Tracker": "https://github.com/markub3327/rl-toolkit/issues",
    },
    download_url="https://github.com/markub3327/rl-toolkit/releases",
    packages=find_packages(),
    keywords=[
        "reinforcement-learning",
        "ml",
        "pybullet",
        "gymnasium",
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
        "gymnasium",
        "box2d-py",
        "pygame",
        "swig",
        "pybullet",
        "dm_control",
        "tensorflow_probability",
        "wandb",
    ],
    extras_require=extras,
    license="MIT License",
    python_requires=">=3.9",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Environment :: Console",
        "Environment :: GPU :: NVIDIA CUDA",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Robotics",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    entry_points={
        "console_scripts": [
            "rl_toolkit = rl_toolkit:main",
        ],
    },
)
