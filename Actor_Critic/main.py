import os
from itertools import count
import torch
import torch.optim as optim
from actor import Actor
from critic import Critic
from grid_world import GridWorldEnv

state_size = 2
action_size = 4
lr = 1e-4

env = GridWorldEnv(width=4, height=3, start=(3, 0), goal=(0, 2))


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


def trainIters(actor, critic, n_iters):
    optimizerA = optim.Adam(actor.parameters())
    optimizerC = optim.Adam(critic.parameters())
    for epoch in range(n_iters):
        state = env.reset()
        log_probs = []
        values = []
        rewards = []
        masks = []
        total_action = []
        entropy = 0
        env.reset()

        for i in count():
            state = torch.FloatTensor(state)
            dist, value = actor(state), critic(state)

            action = dist.sample()
            next_state, reward, done, _ = env.step(action.cpu().numpy())
            total_action.append(env.t_action[action.cpu().numpy()])

            log_prob = dist.log_prob(action).unsqueeze(0)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float))
            masks.append(torch.tensor([1-done], dtype=torch.float))

            state = next_state

            if done:
                print("epoch = {}, tracked actions to attempt goal = {}".format(
                    epoch, total_action))
                break

        next_state = torch.FloatTensor(next_state)
        next_value = critic(next_state)
        returns = compute_returns(next_value, rewards, masks)

        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        optimizerA.zero_grad()
        optimizerC.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        optimizerA.step()
        optimizerC.step()


if __name__ == '__main__':
    actor = Actor(state_size, action_size)
    critic = Critic(state_size, action_size)
    trainIters(actor, critic, n_iters=200)
