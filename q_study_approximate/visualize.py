import time


def print_values(values, width, height):
    """
        print grid values
        Args:
            - values: {(x: int, y: int): reward: int}
            - width: int
            - height: int
    """
    time.sleep(1)
    for j in reversed(range(height)):
        print("'---------------------------------------")
        for i in range(width):
            v = values.get((i, j), 0.0)
            print("| ", '{:06.2f}'.format(v), end=" ")
        print("| ")
    print("'---------------------------------------")
    print("")
    print("")


def print_policies(policies, width, height):
    """
        print grid policy
        Args:
            - policies: {(x: int, y: int): action: str}
            - width: int
            - height: int
    """
    time.sleep(2)
    for j in reversed(range(height)):
        print("-----------------------------")
        for i in range(width):
            a = policies.get((i, j), ' ')
            print("| ", '{:3s}'.format(a), end=" ")
        print("| ")
    print("-----------------------------")
    print("")
    print("")
