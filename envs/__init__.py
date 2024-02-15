from gymnasium.envs.registration import register
from envs.super_mario.mo_super_mario import MOSuperMarioBros
from envs.super_mario.super_mario import SuperMarioBros

register(
    id="mo-LunarLander-v2",
    entry_point="envs.lunar_lander.lunar_lander:LunarLanderStratV2",
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id="LunarLander-v2",
    entry_point="envs.lunar_lander.lunar_lander:LunarLanderStratV2",
    kwargs={"stratified": False},
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id="mo-LunarLanderContinuous-v2",
    entry_point="envs.lunar_lander.lunar_lander:LunarLanderStratV2",
    kwargs={"continuous": True},
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id="LunarLanderContinuous-v2",
    entry_point="envs.lunar_lander.lunar_lander:LunarLanderStratV2",
    kwargs={"continuous": True, "stratified": False},
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id="SuperMarioBros-v0",
    entry_point="envs.super_mario.super_mario:SuperMarioBros",
    nondeterministic=True,
)

register(
    id="mo-SuperMarioBros-v0",
    entry_point="envs.super_mario.mo_super_mario:MOSuperMarioBros",
    nondeterministic=True,
)
