import argparse
import random

from yaml import safe_load

import gymnasium as gym
import mo_gymnasium as mogym
import numpy as np
import torch
import wandb

import envs

# from stable_baselines3.common.atari_wrappers import (
#     MaxAndSkipEnv,
# )
from gymnasium.wrappers import (
    FrameStack,
    GrayScaleObservation,
    ResizeObservation,
    TimeLimit,
)
from mo_gymnasium.envs.mario.joypad_space import JoypadSpace
from mo_gymnasium.utils import MOMaxAndSkipObservation
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


def strtobool(value: str) -> bool:
    value = value.lower()
    if value in ("y", "yes", "on", "1", "true", "t"):
        return True
    return False


def make_env(args, idx, run_name):
    def thunk():
        env = mogym.make(
            args.gym_id,
            render_mode="rgb_array" if args.capture_video and idx == 0 else None,
        )
        if args.capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(
                env,
                f"videos/{run_name}",
                episode_trigger=lambda x: x % args.video_freq == 0,
            )
        if args.stratified:
            env = mogym.MORecordEpisodeStatistics(env)
        else:
            env = mogym.LinearReward(env)
            env = gym.wrappers.RecordEpisodeStatistics(env)
        if args.with_image:
            if "SuperMario" in args.gym_id:
                env = JoypadSpace(env, SIMPLE_MOVEMENT)
            if args.stratified:
                env = MOMaxAndSkipObservation(env, skip=4)
            # else:
            #     env = MaxAndSkipEnv(env, skip=4)
            env = ResizeObservation(env, (84, 84))
            env = GrayScaleObservation(env)
            env = FrameStack(env, 4)
            env = TimeLimit(env, max_episode_steps=1000)
        env.action_space.seed(args.seed)
        return env

    return thunk


def base_hyperparams():
    hyper_params = {
        "with_image": False,
        "stratified": False,
        "seed": 0,
        "total_timesteps": 1000000,
        "torch_deterministic": True,
        "cuda": True,
        "capture_video": False,
        "video_freq": 50,
        "track": False,
        "num_envs": 1,
        "buffer_size": int(1e6),
        "gamma": 0.99,
        "tau": 1 - 5e-3,
        "batch_size": 256,
        "exploration_noise": 0.1,
        "learning_starts": 5e3,
        "policy_lr": 3e-4,
        "q_lr": 1e-3,
        "policy_frequency": 2,
        "target_network_frequency": 1,
        "noise_clip": 0.5,
        "alpha": 0.2,
        "target_entropy_scale": 1,
        "epsilon": 1e-6,
        "autotune": True,
        "reward_scaling": 1.0,
        "num_rewards": 1,
        "dylam_rb": 10,
        "dylam_tau": 0.995,
        "dylam": False,
        "lambdas": [1],
        "r_max": [1],
        "r_min": [0],
        "hidden_dim": 256,
        "ou_noise": False,
    }
    return hyper_params


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", type=str, default="Baseline")
    parser.add_argument("--env", type=str, default="LunarLander")
    parser.add_argument(
        "--cuda",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, cuda will be enabled by default",
    )
    parser.add_argument(
        "--capture-video",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)",
    )
    parser.add_argument(
        "--video-freq",
        type=int,
        default=50,
        help="Frequency of saving videos, in episodes",
    )
    parser.add_argument(
        "--track",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="Log on wandb",
    )
    args = parser.parse_args()
    args.env = args.env.lower().upper()
    args.setup = args.setup.lower().title()
    return args


def get_experiment(arguments):
    with open("experiments.yml", "r") as f:
        params = safe_load(f)
    experiment = base_hyperparams()
    experiment.update(vars(arguments))
    experiment.update(params[arguments.setup][arguments.env])
    experiment = argparse.Namespace(**experiment)
    return experiment


def setup_run(params):
    # TRY NOT TO MODIFY: seeding
    random.seed(params.seed)
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.backends.cudnn.deterministic = params.torch_deterministic
