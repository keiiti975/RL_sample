import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, input_dim=2, output_dim=4):
        """
            init function
            Args:
                - input_dim: int
                - output_dim: int
        """
        super(Actor, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, 3)
        self.fc2 = nn.Linear(3, output_dim)
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
        print("initialize Actor weight ...")
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 1., 1.)
                nn.init.constant_(m.bias, 0.)


class Actor_Loss(nn.Module):
    def __init__(self):
        """
            init function
        """
        super(Actor_Loss, self).__init__()

    def forward(self, pai, a, td_error):
        """
            forward function
            Args:
                - pai: torch.FloatTensor, shape==[1, output_dim]
                - a: int
                - td_error: float
        """
        log_probs = torch.log(pai[0, a] + 1.0)
        loss = torch.mean(log_probs * td_error)
        return loss


def choose_action(pai):
    """
        choose action from pai
        Args:
            - pai: torch.FloatTensor, shape==[1, output_dim]
    """
    probs = F.softmax(pai, dim=-1)
    probs = probs.detach().numpy()
    return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())
