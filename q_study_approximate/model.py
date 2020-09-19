import numpy as np


class Model:
    def __init__(self):
        self.theta = np.random.randn(25) / np.sqrt(25)

    def sa2x(self, s, a):
        return np.array([
            s[1] - 1 if a == 'U' else 0,
            s[0] - 1.5 if a == 'U' else 0,
            (s[0]*s[1] - 3)/3 if a == 'U' else 0,
            (s[1]*s[1] - 2)/2 if a == 'U' else 0,
            (s[0]*s[0] - 4.5)/4.5 if a == 'U' else 0,
            1 if a == 'U' else 0,
            s[1] - 1 if a == 'D' else 0,
            s[0] - 1.5 if a == 'D' else 0,
            (s[1]*s[0] - 3)/3 if a == 'D' else 0,
            (s[1]*s[1] - 2)/2 if a == 'D' else 0,
            (s[0]*s[0] - 4.5)/4.5 if a == 'D' else 0,
            1 if a == 'D' else 0,
            s[1] - 1 if a == 'L' else 0,
            s[0] - 1.5 if a == 'L' else 0,
            (s[0]*s[1] - 3)/3 if a == 'L' else 0,
            (s[1]*s[1] - 2)/2 if a == 'L' else 0,
            (s[0]*s[0] - 4.5)/4.5 if a == 'L' else 0,
            1 if a == 'L' else 0,
            s[1] - 1 if a == 'R' else 0,
            s[0] - 1.5 if a == 'R' else 0,
            (s[0]*s[1] - 3)/3 if a == 'R' else 0,
            (s[1]*s[1] - 2)/2 if a == 'R' else 0,
            (s[0]*s[0] - 4.5)/4.5 if a == 'R' else 0,
            1 if a == 'R' else 0,
            1
        ])

    def predict(self, s, a):
        x = self.sa2x(s, a)
        return self.theta.dot(x)

    def grad(self, s, a):
        return self.sa2x(s, a)
