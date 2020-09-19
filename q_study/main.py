import numpy as np
from grid_world import standard_grid, negative_grid
from visualize import print_values, print_policies
from q_study_functional import action_eps_greedy, get_max_Q

GAMMA = 0.9
ALPHA = 0.1
ALL_POSSIBLE_ACTIONS = ('R', 'L', 'U', 'D')

if __name__ == "__main__":
    grid = standard_grid()
    print_values(grid.rewards, grid.width, grid.height)
    Q = {}
    states = grid.get_all_states()
    for s in states:
        Q[s] = {}
        for a in ALL_POSSIBLE_ACTIONS:
            Q[s][a] = 0
    t = 1.0
    for it in range(100):
        if it % 100 == 0:
            t += 1e-2
        # 0. initialize condition
        current_grid = (3, 0)
        grid.set_initial_grid(current_grid)
        start_flag = True
        while not grid.game_over():
            if start_flag is True:
                # 1. choose random action
                action = np.random.choice(ALL_POSSIBLE_ACTIONS)
            else:
                # 1 + 4n (n = 1, 2, ...). choose action by eps_greedy
                current_Q = Q[current_grid]
                action = action_eps_greedy(
                    current_Q, ALL_POSSIBLE_ACTIONS, eps=0.5/t)
            # 2 + 4n (n = 0, 1, ...). update condition
            r = grid.move(action)
            s2 = grid.current_grid()
            # 3 + 4n (n = 0, 1, ...). get max Q
            next_Q = Q[s2]
            a2, max_q_s2a2 = get_max_Q(next_Q)
            # 4 + 4n (n = 0, 1, ...). update Q
            Q[current_grid][action] = Q[current_grid][action] + \
                ALPHA*(r + GAMMA*max_q_s2a2 - Q[current_grid][action])
            start_flag = False
            current_grid = s2

    policy = {}
    V = {}
    for s in grid.actions.keys():
        a, max_q = get_max_Q(Q[s])
        policy[s] = a
        V[s] = max_q

    print("values:")
    print_values(V, grid.width, grid.height)
    print("policy:")
    print_policies(policy, grid.width, grid.height)
