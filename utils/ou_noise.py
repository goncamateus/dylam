import numpy as np


class OrnsteinUhlenbeckNoise:
    """Add Ornstein-Uhlenbeck noise to continuous actions.

    https://en.wikipedia.org/wiki/Ornstein-Uhlenbeck_process_

    Parameters
    ----------
    mu : float or ndarray, optional

        The mean towards which the Ornstein-Uhlenbeck process should revert; must be
        broadcastable with the input actions.

    sigma : positive float or ndarray, optional

        The spread of the noise of the Ornstein-Uhlenbeck process; must be
        broadcastable with the input actions.

    theta : positive float or ndarray, optional

        The (element-wise) dissipation rate of the Ornstein-Uhlenbeck process; must
        be broadcastable with the input actions.

    min_value : float or ndarray, optional

        The lower bound used for clipping the output action; must be broadcastable with the input
        actions.

    max_value : float or ndarray, optional

        The upper bound used for clipping the output action; must be broadcastable with the input
        actions.

    random_seed : int, optional

        Sets the random state to get reproducible results.
    """

    def __init__(
        self,
        mu=0.0,
        sigma=1.0,
        theta=0.15,
        min_value=None,
        max_value=None,
        random_seed=None,
    ):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.min_value = min_value
        self.max_value = max_value
        self.random_state = np.random.RandomState(random_seed)
        self.reset()
    
    def reset(self):
        """Reset the Ornstein-Uhlenbeck process."""
        self.state = self.mu
        
    def __call__(self, action):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * self.random_state.randn(len(x))
        self.state = x + dx
        action += self.state
        if self.min_value is not None:
            action = np.maximum(action, self.min_value)
        if self.max_value is not None:
            action = np.minimum(action, self.max_value)
        return action
