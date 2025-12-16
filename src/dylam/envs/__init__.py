from gymnasium.envs.registration import register

register(
    id="mo-LunarLanderContinuous-v2",
    entry_point="dylam.envs.lunar_lander:LunarLanderStrat",
    kwargs={"continuous": True},
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id="mo-HalfCheetah-v4",
    entry_point="dylam.envs.half_cheetah:HalfCheetah",
    max_episode_steps=1000,
)

register(
    id="mo-VSS-v0",
    entry_point="dylam.envs.vss:VSSStratEnv",
    max_episode_steps=400,
)

register(
    id="mo-Taxi-v3",
    entry_point="dylam.envs.taxi:Taxi",
    max_episode_steps=200,
)

register(
    id="mo-LunarLander-v2",
    entry_point="dylam.envs.lunar_lander:LunarLanderStrat",
    kwargs={"continuous": False},
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id="mo-Minecart-v0",
    entry_point="dylam.envs.minecart:MinecartEnv",
    max_episode_steps=1000,
)

register(
    id="mo-ChickenBanana-v0",
    entry_point="dylam.envs.chicken_banana:ChickenBanana",
    max_episode_steps=80,
)
