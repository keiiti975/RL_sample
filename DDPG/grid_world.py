import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class GridWorldEnv(gym.Env):
    """
    Description:
        search start to goal route from grid world
    Source:

    Observation:
        Type: Box(2)
        Num   Observation   Min   Max
        0     X             0     width
        1     Y             0     height
    Actions:
        Type: Discrete(4)
        Num   Action
        0     go right
        1     go left
        2     go up
        3     go down
    Reward:
        -10 per step,
        100 for reach the goal (0, 2)
    Starting State:
        (3, 0)
    """

    def __init__(self, width=4, height=3, start=(3, 0), goal=(0, 2)):
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        """
        self.t_state = {
            ...     ...     ...
            (0, 2), (1, 2), (2, 2), ...
            (0, 1), (1, 1), (2, 1), ...
            (0, 0), (1, 0), (2, 0), ...
        }
        """
        self.t_action = ['R', 'L', 'U', 'D']
        self.actions = {
            (0, 0): ('R', 'U'),
            (1, 0): ('R', 'L', 'U'),
            (1, 1): ('L', 'U', 'D'),
            (1, 2): ('R', 'L', 'D'),
            (2, 0): ('R', 'L'),
            (2, 2): ('R', 'L'),
            (3, 0): ('L', 'U'),
            (3, 1): ('U', 'D'),
            (3, 2): ('L', 'D'),
        }
        rewards = {
            (0, 1): -100,
            (0, 2): 100
        }
        self.set_rewards(rewards, step_cost=-10)
        self.init_render()

        # search space
        low = np.array([0, 0], dtype=np.int32)
        high = np.array([width, height], dtype=np.int32)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low, high, dtype=np.int32)
        self.seed()

    def set_rewards(self, rewards, step_cost=-10):
        self.rewards = {}
        for j in range(self.height):
            for i in range(self.width):
                self.rewards.update({(i, j): step_cost})
        self.rewards.update(rewards)

    def move(self, action):
        # check if legal move first
        state = list(self.state)
        if action in self.actions[self.state]:
            if action == 'R':
                state[0] += 1
            elif action == 'L':
                state[0] -= 1
            elif action == 'U':
                state[1] += 1
            elif action == 'D':
                state[1] -= 1
        self.state = tuple(state)
        return self.rewards.get(self.state, 0)

    def game_over(self):
        return self.state not in self.actions

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        reward = self.move(self.t_action[action])
        done = self.game_over()
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.start
        return np.array(self.state)

    def init_render(self):
        print("----- grid world problem -----")
        print()
        grid_char = []
        for y in range(self.height):
            for x in range(self.width):
                if x == self.start[0] and y == self.start[1]:
                    grid_char.append("S")
                elif x == self.goal[0] and y == self.goal[1]:
                    grid_char.append("G")
                else:
                    grid_char.append(" ")

        for y in reversed(range(self.height)):
            print("."*self.width*4)
            for x in range(self.width):
                if x == self.width - 1:
                    print("| " + grid_char[self.width * y + x] + " |")
                else:
                    print("| " + grid_char[self.width * y + x] + " ", end="")
        print("."*self.width*4)
        print()
        print("------------------------------")
