import numpy as np


def action_eps_greedy(current_Q, ALL_POSSIBLE_ACTIONS, eps=0.1):
    """
        sample action by epsilon greedy policy
        Args:
            - current_Q: {'R': float, 'L': float, 'U': float, 'D': float}
            - eps: float
    """
    # get action greedy
    key_list = np.array(list(current_Q.keys()))
    value_list = np.array(list(current_Q.values()))
    max_value_index = np.argmax(value_list)
    action_greedy = key_list[max_value_index]
    # get action eps greedy
    p = np.random.random()
    if p < eps:
        list_ALL_POSSIBLE_ACTIONS = [elem for elem in list(
            ALL_POSSIBLE_ACTIONS) if elem != action_greedy]
        return np.random.choice(list_ALL_POSSIBLE_ACTIONS)
    else:
        return action_greedy


def get_max_Q(Q):
    """
        return max Q, key and value
        Args:
            - Q: {'R': float, 'L': float, 'U': float, 'D': float}
    """
    max_key = None
    max_val = float('-inf')
    for k, v in Q.items():
        if v > max_val:
            max_key = k
            max_val = v
    return max_key, max_val
