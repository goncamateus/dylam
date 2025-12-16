# DyLam: A Dynamic Reward Weighting Framework for Reinforcement Learning Algorithms

This repository contains the code for the paper "[DyLam: A Dynamic Reward Weighting Framework for Reinforcement Learning Algorithms](https://ifaamas.csc.liv.ac.uk/Proceedings/aamas2025/pdfs/p2651.pdf)" by [Mateus Machado](https://github.com/goncamateus), and [Hansenclever Bassani](https://hfbassani.github.io/).

## Installation

To install the dependencies:

```bash
pip install pipx
pipx install uv
uv venv
uv pip install .
```

Then, run the container:

```bash
python train.py --env [GYM-ID] --setup [SETUP] --capture-video --track
```
