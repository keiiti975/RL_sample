import numpy as np


class OUNoise:
    def __init__(self, action_size, mu=0., theta=0.15, sigma=0.2):
        """
            init function  
            Args:  
                - action_size: int  
                - mu: float  
                - theta: float  
                - sigma: float  
        """
        self.action_size = action_size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_size) * self.mu
        self.reset()

    def reset(self):
        """
            initialize noise  
        """
        self.state = np.ones(self.action_size) * self.mu

    def noise(self):
        """
            generate noise  
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * \
            np.random.randn(self.action_size)
        self.state = x + dx
        return self.state
