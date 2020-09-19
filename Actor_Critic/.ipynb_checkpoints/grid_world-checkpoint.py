import numpy as np


class Grid:  # Environment
    def __init__(self):
        self.width = 4
        self.height = 3
        self.i = 4
        self.j = 1
        self.start = [4, 1]
        self.goal = [1, 3]
        self.t_state = {
            (1, 3), (2, 3), (3, 3), (4, 3),
            (1, 2), (2, 2), (3, 2), (4, 2),
            (1, 1), (2, 1), (3, 1), (4, 1),
        }
        self.t_action = ['R', 'L', 'U', 'D']
        self.actions = {
            (1, 1): ('R', 'U'),
            (2, 1): ('R', 'L', 'U'),
            (2, 2): ('L', 'U', 'D'),
            (2, 3): ('R', 'L', 'D'),
            (3, 1): ('R', 'L'),
            (3, 3): ('R', 'L'),
            (4, 1): ('L', 'U'),
            (4, 2): ('U', 'D'),
            (4, 3): ('L', 'D'),
        }
        self.rewards = {
            (1, 2): -100,
            (1, 3): 100,
        }

    def reset(self):
        self.__init__()
        return [self.i, self.j]

    def step(self, action):
        Reward = self.move(action)
        state = [self.i, self.j]
        terminate = self.game_over()
        if terminate:
            self.reset()
        return state, Reward, terminate

    def current_state(self):
        return [self.i, self.j]

    def is_terminal(self, s):
        return s not in self.actions

    def move(self, action):
        # check if legal move first
        if action in self.actions[(self.i, self.j)]:
            if action == 'R':
                self.i += 1
            elif action == 'L':
                self.i -= 1
            elif action == 'U':
                self.j += 1
            elif action == 'D':
                self.j -= 1
        return self.rewards.get((self.i, self.j), 1)

    def undo_move(self, action):
        if action == 'R':
            self.i -= 1
        elif action == 'L':
            self.i += 1
        elif action == 'U':
            self.j -= 1
        elif action == 'D':
            self.j += 1
        assert(self.current_state() in self.all_states())

    def game_over(self):
        return (self.i, self.j) not in self.actions

    def all_states(self):
        return set(self.actions.keys() + self.rewards.keys())

    def draw_board(self):
        board = []
        for J in range(self.height):
            for I in range(self.width):
                if I == 3 and J == 0:
                    board.append("S")
                elif I == 0 and J == 2:
                    board.append("G")
                else:
                    board.append(" ")
        print(" "*15, ".....................")
        print(" "*15, "|", "".join(board[8]), " |", "".join(board[9]),
              " |", "".join(board[10]), " |", "".join(board[11]), " |")
        print(" "*15, "|----|----|----|----|")
        print(" "*15, "|", "".join(board[4]), " |", "".join(board[5]),
              " |", "".join(board[6]), " |", "".join(board[7]), " |")
        print(" "*15, "|----|----|----|----|")
        print(" "*15, "|", "".join(board[0]), " |", "".join(board[1]),
              " |", "".join(board[2]), " |", "".join(board[3]), " |")

        print(" "*15, "''''''''''''''''''''")
