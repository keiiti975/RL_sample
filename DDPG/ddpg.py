import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from buffer import ReplayBuffer
from random_process import OUNoise
from model import Actor, Critic
from utils import hard_update, soft_update


class DDPG:
    def __init__(self, args):
        """
            init function
            Args:
                - args: class with args parameter
        """
        self.state_size = args.state_size
        self.action_size = args.action_size
        self.bs = args.bs
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.tau = args.tau
        self.discrete = args.discrete
        self.randomer = OUNoise(args.action_size)
        self.buffer = ReplayBuffer(args.max_buff)

        self.actor = Actor(self.state_size, self.action_size)
        self.actor_target = Actor(self.state_size, self.action_size)
        self.actor_opt = AdamW(self.actor.parameters(), args.lr_actor)

        self.critic = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)
        self.critic_opt = AdamW(self.critic.parameters(), args.lr_critic)

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

    def reset(self):
        """
            reset noise and model
        """
        self.randomer.reset()

    def get_action(self, state):
        """
            get distribution of action
            Args:
                - state: list, shape == [state_size]
        """
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        action = self.actor(state).detach()
        action = action.squeeze(0).numpy()
        action += self.epsilon * self.randomer.noise()
        action = np.clip(action, -1.0, 1.0)
        return action

    def learning(self):
        """
            learn models
        """
        s1, a1, r1, t1, s2 = self.buffer.sample_batch(self.bs)
        # bool -> int
        t1 = 1 - t1
        s1 = torch.tensor(s1, dtype=torch.float)
        a1 = torch.tensor(a1, dtype=torch.float)
        r1 = torch.tensor(r1, dtype=torch.float)
        t1 = torch.tensor(t1, dtype=torch.float)
        s2 = torch.tensor(s2, dtype=torch.float)

        a2 = self.actor_target(s2).detach()
        q2 = self.critic_target(s2, a2).detach()
        q2_plus_r = r1[:, None] + t1[:, None] * self.gamma * q2
        q1 = self.critic.forward(s1, a1)

        # critic gradient
        critic_loss = nn.MSELoss()
        loss_critic = critic_loss(q1, q2_plus_r)
        self.critic_opt.zero_grad()
        loss_critic.backward()
        self.critic_opt.step()

        # actor gradient
        pred_a = self.actor.forward(s1)
        loss_actor = (-self.critic.forward(s1, pred_a)).mean()
        self.actor_opt.zero_grad()
        loss_actor.backward()
        self.actor_opt.step()

        # Notice that we only have gradient updates for actor and critic, not target
        # actor_opt.step() and critic_opt.step()
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return loss_actor.item(), loss_critic.item()
