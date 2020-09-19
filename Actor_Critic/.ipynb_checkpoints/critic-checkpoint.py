import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, input_dim=2):
        """
            init function
            Args:
                - input_dim: int
        """
        super(Critic, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, 20)
        self.fc2 = nn.Linear(20, 1)
        self.relu = nn.ReLU(inplace=True)
        self._initialize_weights()

    def forward(self, s):
        """
            forward function
            Args:
                - s: torch.FloatTensor, shape==[1, input_dim]
        """
        s = self.relu(self.fc1(s))
        s = self.relu(self.fc2(s))
        return s

    def _initialize_weights(self):
        print("initialize Critic weight ...")
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 1., 1.)
                nn.init.constant_(m.bias, 0.)


class Critic_Loss(nn.Module):
    def __init__(self):
        """
            init function
        """
        super(Critic_Loss, self).__init__()
        self.GAMMA = 0.98

    def forward(self, r, v_, v):
        """
            forward function
            Args:
                - r: float
                - v_: torch.FloatTensor, shape==[1, 1]
                - v: float, shape==[1, 1]
        """
        td_error = r + torch.sum(self.GAMMA * v_ - v)
        return td_error
