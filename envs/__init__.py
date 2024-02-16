from gymnasium.envs.registration import register

register(
    id="mo-LunarLander-v2",
    entry_point="envs.lunar_lander.lunar_lander:LunarLanderStratV2",
    kwargs={"stratified": True},
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
    kwargs={"continuous": True, "stratified": True},
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
    id="mo-MountainCar-v0",
    entry_point="envs.mountain_car.discrete:MountainCar",
    kwargs={"stratified": True},
    max_episode_steps=999,
)

register(
    id="MountainCar-v0",
    entry_point="envs.mountain_car.discrete:MountainCar",
    kwargs={"stratified": False},
    max_episode_steps=999,
)

register(
    id="mo-MountainCarContinuous-v0",
    entry_point="envs.mountain_car.continuous:ContinuousMountainCar",
    kwargs={"stratified": True},
    max_episode_steps=999,
)

register(
    id="MountainCarContinuous-v0",
    entry_point="envs.mountain_car.continuous:ContinuousMountainCar",
    kwargs={"stratified": False},
    max_episode_steps=999,
)

register(
    id="mo-Hopper-v4",
    entry_point="envs.hopper.hopper:Hopper",
    kwargs={"stratified": True},
    max_episode_steps=1000,
)

register(
    id="Hopper-v4",
    entry_point="envs.hopper.hopper:Hopper",
    kwargs={"stratified": False},
    max_episode_steps=1000,
)

register(
    id="mo-SuperMario-v0",
    entry_point="envs.super_mario.super_mario:SuperMarioBros",
    kwargs={"stratified": True},
    nondeterministic=True,
)

register(
    id="SuperMario-v0",
    entry_point="envs.super_mario.super_mario:SuperMarioBros",
    kwargs={"stratified": False},
    nondeterministic=True,
)
