from gymnasium.envs.registration import register


register(
    id="mo-LunarLanderContinuous-v2",
    entry_point="envs.lunar_lander:LunarLanderStrat",
    kwargs={"continuous": True},
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id="mo-Hopper-v4",
    entry_point="envs.hopper:Hopper",
    max_episode_steps=1000,
)


register(
    id="mo-Pendulum-v1",
    entry_point="envs.pendulum:Pendulum",
    max_episode_steps=200,
)

register(
    id="mo-VSS-v0",
    entry_point="envs.vss:VSS",
    max_episode_steps=200,
)