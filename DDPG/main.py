import numpy as np
from ddpg import DDPG
from gym import spaces
from grid_world import GridWorldEnv

env = GridWorldEnv(width=4, height=3, start=(3, 0), goal=(0, 2))


class args:
    # training config
    gamma = 0.999
    episodes = 800
    bs = 20
    lr_actor = 1e-3
    lr_critic = 1e-3
    epsilon = 1.0
    tau = 1e-2
    max_steps = 1e2
    max_buff = 1e6
    # experiment config
    if isinstance(env.action_space, spaces.Discrete):
        action_size = env.action_space.n
        discrete = True
    elif isinstance(env.action_space, spaces.Box):
        action_size = int(env.action_space.shape[0])
        discrete = False
    else:
        print("error! only Discrete or Box is allowed in action_space")

    if isinstance(env.observation_space, spaces.Discrete):
        state_size = env.observation_space.n
    elif isinstance(env.observation_space, spaces.Box):
        state_size = int(env.observation_space.shape[0])
    else:
        print("error! only Discrete or Box is allowed in observation_space")


def train(agent, env, episodes=800, bs=20, max_steps=100):
    all_rewards = []
    for episode in range(episodes):
        s1 = env.reset()
        agent.reset()

        done = False
        step = 0
        total_action = []

        while not done:
            action = agent.get_action(s1)

            if args.discrete is True:
                s2, r1, done, _ = env.step(np.argmax(action))
            else:
                s2, r1, done, _ = env.step(action)
            agent.buffer.add(s1, action, r1, done, s2)
            s1 = s2
            total_action.append(env.t_action[np.argmax(action)])

            if agent.buffer.size() > bs:
                loss_a, loss_c = agent.learning()

            step += 1

            if step + 1 > max_steps:
                break

        print("episode = {}, tracked actions to attempt goal = {}".format(
            episode, total_action))


if __name__ == '__main__':
    agent = DDPG(args)
    train(agent, env, episodes=args.episodes,
          bs=args.bs, max_steps=args.max_steps)
