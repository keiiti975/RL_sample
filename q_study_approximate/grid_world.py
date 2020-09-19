import numpy as np


class Grid:
    def __init__(self, width, height, start):
        """
            Grid World initializer
            Args:
                - width: int
                - height: int
                - start: (int, int)
        """
        self.width = width
        self.height = height
        self.x = start[0]
        self.y = start[1]

    def set_condition(self, rewards, actions):
        """
            set reward and action
            Args:
                rewards: {(x: int, y: int): reward: int}
                actions: {(x: int, y: int): (action: str, ...)}
                    action in ["R", "L", "U", "D"]
        """
        self.rewards = rewards
        self.actions = actions

    def set_initial_grid(self, initial_grid):
        """
            set initial grid
            Args:
                - initial_grid: (int, int)
        """
        self.x = initial_grid[0]
        self.y = initial_grid[1]

    def get_all_states(self):
        """
            return all grid key
        """
        return set(list(self.actions.keys()) + list(self.rewards.keys()))

    def game_over(self):
        """
            return game over or not
            if (x, y) not in grid, return True
            else, return False
        """
        return (self.x, self.y) not in self.actions

    def move(self, action):
        """
            move grid
            Args:
                - action: str
        """
        if action in self.actions[(self.x, self.y)]:
            if action == 'R':
                self.x += 1
            elif action == 'L':
                self.x -= 1
            elif action == 'U':
                self.y += 1
            elif action == 'D':
                self.y -= 1
        return self.rewards.get((self.x, self.y), 0.0)

    def current_grid(self):
        """
            return current grid
        """
        return (self.x, self.y)

    def is_terminal(self, s):
        """
            check s in grid
            Args:
                s: (x: int, y: int)
        """
        return s not in self.actions

    def undo_move(self, action):
        if action == 'R':
            self.y -= 1
        elif action == 'L':
            self.y += 1
        elif action == 'U':
            self.x -= 1
        elif action == 'D':
            self.x += 1
        assert(self.current_state() in self.all_states())


def standard_grid():
    """
        create standard grid
        if you change config, overwrite this function directly
    """
    # grid config
    width = 4
    height = 3
    start = (3, 1)
    rewards = {
        (0, 1): -100,
        (0, 2): 100
    }
    actions = {
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
    # create grid world
    g = Grid(width, height, start)
    g.set_condition(rewards, actions)
    return g


def negative_grid(step_cost=-20):
    g = standard_grid()
    g.rewards.update({
        (3, 2): step_cost,
        (2, 2): step_cost,
        (3, 1): step_cost,
        (1, 1): step_cost,
        (1, 2): step_cost,
        (0, 0): step_cost,
        (1, 0): step_cost,
        (2, 0): step_cost,
        (3, 0): step_cost,
    })
    return g
