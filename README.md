# DyLam: A Dynamic Reward Weighting Framework for Reinforcement Learning Algorithms

This repository contains the code for the paper "[DyLam: A Dynamic Reward Weighting Framework for Reinforcement Learning Algorithms](https://google.com)" by [Mateus Machado](https://github.com/goncamateus), and [Hansenclever Bassani](https://hfbassani.github.io/).

## Installation (Docker)

To install the dependencies, you can use the provided Dockerfile. First, build the image:

```bash
docker build -t dylam .
```

Then, run the container:

```bash
docker run -it dylam poetry run python train.py --env [GYM-ID] --setup [SETUP] --capture-video --track
```