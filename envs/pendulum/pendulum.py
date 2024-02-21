from gymnasium.envs.classic_control.pendulum import PendulumEnv

class PendulumStrat(PendulumEnv):
    def __init__(self, stratified: bool = False):
        super().__init__()
        self.stratified = stratified

    def step(self, action):
        if self.stratified:
            action = action[0]
        return super().step(action)

    def reset(self):
        return super().reset()