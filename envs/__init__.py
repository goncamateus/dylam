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
    id="mo-Humanoid-v4",
    entry_point="envs.humanoid:Humanoid",
    max_episode_steps=1000,
)

register(
    id="mo-HalfCheetah-v4",
    entry_point="envs.half_cheetah:HalfCheetah",
    max_episode_steps=1000,
)

register(
    id="mo-HalfCheetahEF-v4",
    entry_point="envs.half_cheetah:HalfCheetahEfficiency",
    max_episode_steps=1000,
)

register(
    id="mo-VSS-v0",
    entry_point="envs.vss:VSSStratEnv",
    max_episode_steps=400,
)

register(
    id="mo-VSSEF-v0",
    entry_point="envs.vss:VSSEF",
    max_episode_steps=400,
)


register(
    id="mo-Taxi-v3",
    entry_point="envs.taxi:Taxi",
    max_episode_steps=200,
)

register(
    id="mo-MountainCar-v0",
    entry_point="envs.mountain_car:MountainCar",
     max_episode_steps=200,
)

register(
    id="mo-LunarLander-v2",
    entry_point="envs.lunar_lander:LunarLanderStrat",
    kwargs={"continuous": False},
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id="mo-Minecart-v0",
    entry_point="envs.minecart:MinecartEnv",
    max_episode_steps=1000,
)