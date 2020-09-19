import numpy as np
import torch
from torch.optim import AdamW
from grid_world import Grid
from actor import Actor, Actor_Loss, choose_action
from critic import Critic, Critic_Loss

np.random.seed(1)

# training config
MAX_EPISODE = 450
Actor_lr = 1e-3
Critic_lr = 1e-3

# problem setting
grid = Grid()
grid.draw_board()
state_dim = 2
action_dim = 4

# init models
actor = Actor(input_dim=state_dim, output_dim=action_dim)
critic = Critic(input_dim=state_dim)
actor_opt = AdamW(actor.parameters(), lr=Actor_lr)
critic_opt = AdamW(critic.parameters(), lr=Critic_lr)

# init loss
a_loss = Actor_Loss()
c_loss = Critic_Loss()

for i_episode in range(MAX_EPISODE):
    s = grid.reset()
    t = 0
    total_action = []
    done = False
    while(not done and t < 200):
        # step 1
        s = torch.Tensor(s)
        pai = actor(s[None, :])
        # step 2
        a = choose_action(pai)
        # step 3
        s_, r, done = grid.step(grid.t_action[a])
        total_action.append(grid.t_action[a])
        if done:
            r = -200
        # step 4
        s_ = torch.Tensor(s_)
        v = critic(s[None, :])
        v_ = critic(s_[None, :])
        # step 5, 6
        critic_loss = c_loss(r, v_, v)
        # step 7
        pass
        # step 8
        actor_loss = a_loss(pai, a, critic_loss)
        # step 9
        pass
        # step 10
        actor_loss.backward()
        actor_opt.step()
        # step 11, 12
        critic_loss.backward()
        critic_opt.step()
        # other
        s = s_
        t += 1
    print("episode:", i_episode, "  tracked actions to attempt goal:", total_action)
