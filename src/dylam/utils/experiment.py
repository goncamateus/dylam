import argparse
import random

import gymnasium as gym
import mo_gymnasium as mogym
import numpy as np
import torch
from gymnasium.spaces import Dict
from minigrid.wrappers import ImgObsWrapper
from yaml import safe_load


def strtobool(value: str) -> bool:
    value = value.lower()
    if value in ("y", "yes", "on", "1", "true", "t"):
        return True
    return False


def softmax_norm(dQ):
    expdQ = torch.exp(dQ) - 1
    return expdQ / (torch.sum(expdQ, 0) + 1e-4)


def minmax_norm(dQ):
    return torch.clamp((dQ.max() - dQ) / (dQ.max() - dQ.min()), 0, 1)


def l1_norm(dQ):
    return dQ / torch.norm(dQ, p=1)


def make_env(args, idx, run_name):
    def thunk():
        env = mogym.make(
            args.gym_id,
            render_mode="rgb_array" if args.capture_video and idx == 0 else None,
        )
        if isinstance(env.observation_space, Dict):
            env = ImgObsWrapper(env)
        if args.capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(
                env,
                f"videos/{run_name}",
                episode_trigger=lambda x: x % args.video_freq == 0,
            )
        if args.stratified:
            env = mogym.MORecordEpisodeStatistics(env)
        else:
            try:
                env = mogym.LinearReward(env)
            except Exception as e:
                print("Running root (gymansium) env")
                pass
            env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(args.seed)
        return env

    return thunk


def q_make_env(args, run_name):
    env = mogym.make(
        args.gym_id,
        render_mode="rgb_array" if args.capture_video else None,
    )
    if args.capture_video:
        env = gym.wrappers.RecordVideo(
            env,
            f"videos/{run_name}",
            episode_trigger=lambda x: x % args.video_freq == 0,
        )
    if not args.stratified:
        env = mogym.LinearReward(env)
    env.action_space.seed(args.seed)
    return env


def base_hyperparams():
    hyper_params = {
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
        "learning_rate": 2.5e-4,
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
        "normalizer": "softmax",
        "dylam": False,
        "lambdas": [1],
        "r_max": [1],
        "r_min": [0],
        "n_hidden": 1,
        "update_frequency": 1,
        "num_eval_episodes": 10,
        "steps_per_iteration": 1000,
        "strategy": 1,
        "epsilon_decay_factor": 0.97,
        "softmax_temperature": 1,
        "total_episodes": 1000,
        "comp_names": [],
        "capql_angle": 22.5,
        "max_weight": [],
        "min_weight": [],
        "weight_sampler": "angle",
        "weight_sampler_std": 0,
        "anneal_lr": False,
        "num_minibatches": 4,
        "num_steps": 1024,
        "entropy_coef": 0,
        "clip_qf_loss": 0,
        "clip_coef": 0,
        "max_grad_norm": 0,
        "norm_adv": 0,
        "qf_coef": 0,
        "target_kl": 0,
        "gae_lambda": 0,
        "update_epochs": 10,
        "realistic": True,
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
