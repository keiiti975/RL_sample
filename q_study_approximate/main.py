import numpy as np
from grid_world import standard_grid, negative_grid
from visualize import print_values, print_policies
from q_study_functional import action_eps_greedy, get_max_Q, get_Qs
from model import Model

GAMMA = 0.9
ALPHA = 0.1
ALL_POSSIBLE_ACTIONS = ('R', 'L', 'U', 'D')

if __name__ == "__main__":
    grid = negative_grid(step_cost=-50)
    print("rewards:")
    print_values(grid.rewards, grid.width, grid.height)
    model = Model()
    t = 1.0
    for it in range(30000):
        if it % 100 == 0:
            t += 1e-2
        if it % 1000 == 0:
            print("it: {}".format(it))
        # 0. initialize condition
        alpha = ALPHA / t
        current_grid = (3, 0)
        grid.set_initial_grid(current_grid)
        start_flag = True
        Qs = get_Qs(model, current_grid, ALL_POSSIBLE_ACTIONS)
        while not grid.game_over():
            if start_flag is True:
                # 1. choose random action
                action = np.random.choice(ALL_POSSIBLE_ACTIONS)
            else:
                # 1 + 6n (n = 1, 2, ...). choose action by eps_greedy
                current_Qs = get_Qs(model, current_grid, ALL_POSSIBLE_ACTIONS)
                action = action_eps_greedy(
                    current_Qs, ALL_POSSIBLE_ACTIONS, eps=0.5/t)
            # 2 + 6n (n = 0, 1, ...). calculate q_current, predict directly
            # 3 + 6n (n = 0, 1, ...). update condition
            r = grid.move(action)
            s2 = grid.current_grid()
            # 4 + 6n (n = 0, 1, ...). calculate Q_s2
            Q_s2 = get_Qs(model, s2, ALL_POSSIBLE_ACTIONS)
            a2, max_q_s2a2 = get_max_Q(Q_s2)

            if grid.is_terminal(s2):
                model.theta += alpha * \
                    (r - model.predict(current_grid, action)) * \
                    model.grad(current_grid, action)
            else:
                # 5 + 6n (n = 0, 1, ...). calculate loss
                loss = r + GAMMA*max_q_s2a2 - \
                    model.predict(current_grid, action)
                # 6 + 6n (n = 0, 1, ...). update model
                model.theta += alpha * loss * model.grad(current_grid, action)
                start_flag = False
                current_grid = s2

    policy = {}
    V = {}
    Q = {}
    for s in grid.actions.keys():
        Qs = get_Qs(model, s, ALL_POSSIBLE_ACTIONS)
        Q[s] = Qs
        a, max_q = get_max_Q(Qs)
        policy[s] = a
        V[s] = max_q

    print("values:")
    print_values(V, grid.width, grid.height)
    print("policy:")
    print_policies(policy, grid.width, grid.height)
