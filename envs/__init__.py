from gymnasium.envs.registration import register

register(
    id="mo-LunarLander-v2",
    entry_point="envs.lunar_lander.lunar_lander:LunarLanderStrat",
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id="mo-LunarLanderContinuous-v2",
    entry_point="envs.lunar_lander.lunar_lander:LunarLanderStrat",
    kwargs={"continuous": True},
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id="mo-MountainCar-v0",
    entry_point="envs.mountain_car.discrete:MountainCar",
    max_episode_steps=999,
)

register(
    id="mo-MountainCarContinuous-v0",
    entry_point="envs.mountain_car.continuous:ContinuousMountainCar",
    max_episode_steps=999,
)

register(
    id="mo-Hopper-v4",
    entry_point="envs.hopper.hopper:Hopper",
    max_episode_steps=1000,
)

register(
    id="mo-SuperMario-v0",
    entry_point="envs.super_mario.super_mario:SuperMarioBros",
    nondeterministic=True,
)

register(
    id="mo-Pendulum-v1",
    entry_point="envs.pendulum.pendulum:Pendulum",
    max_episode_steps=200,
)
