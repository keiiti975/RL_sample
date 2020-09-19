import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        """
            init function  
            Args:  
                - state_size: int  
                - action_size: int  
        """
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, self.action_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.weight_init()

    def forward(self, state):
        """
            forward function  
            Args:  
                - state: torch.FloatTensor, shape==[batch_size, state_size]  
        """
        output = self.relu(self.linear1(state))
        output = self.relu(self.linear2(output))
        action = self.tanh(self.linear3(output))
        return action

    def weight_init(self):
        """
            initialize weight
        """
        nn.init.normal_(self.linear1.weight, 0, 0.01)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.normal_(self.linear2.weight, 0, 0.01)
        nn.init.constant_(self.linear2.bias, 0)
        nn.init.normal_(self.linear3.weight, 0, 0.01)
        nn.init.constant_(self.linear3.bias, 0)


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        """
            init function  
            Args:  
                - state_size: int  
                - action_size: int  
        """
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size + self.action_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.weight_init()

    def forward(self, state, action):
        """
            forward function  
            Args:  
                - state: torch.FloatTensor, shape==[batch_size, state_size]  
                - action: torch.FloatTensor, shape==[batch_size, action_size]  
        """
        model_input = torch.cat([state, action], dim=1)
        output = self.relu(self.linear1(model_input))
        output = self.relu(self.linear2(output))
        value = self.linear3(output)
        return value

    def weight_init(self):
        """
            initialize weight
        """
        nn.init.normal_(self.linear1.weight, 0, 0.01)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.normal_(self.linear2.weight, 0, 0.01)
        nn.init.constant_(self.linear2.bias, 0)
        nn.init.normal_(self.linear3.weight, 0, 0.01)
        nn.init.constant_(self.linear3.bias, 0)
