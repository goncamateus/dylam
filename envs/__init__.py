from gymnasium.envs.registration import register
from envs.super_mario.mo_super_mario import MOSuperMarioBros
from envs.super_mario.super_mario import SuperMarioBros

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
