import random
import numpy as np


class ReplayBuffer(object):
    def __init__(self, buffer_size, random_seed=123):
        """
            init function  
            Args:  
                - buffer_size: int  
                - random_seed: int  
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = []
        random.seed(random_seed)

    def add(self, s1, a1, r1, t1, s2):
        """
            add observed parameters to buffer  
            Args:  
                s1: np.array, shape==[state_size]  
                a1: np.array, shape==[action_size]  
                r1: int  
                t1: bool  
                s2: np.array, shape==[state_size]  
        """
        experience = (s1, a1, r1, t1, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.pop(0)
            self.buffer.append(experience)

    def size(self):
        """
            return count of buffer  
        """
        return self.count

    def sample_batch(self, batch_size):
        """
            sample batch data from buffer  
            Args:  
                - batch_size: int  
        """
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s1_batch = np.array([_[0] for _ in batch])
        a1_batch = np.array([_[1] for _ in batch])
        r1_batch = np.array([_[2] for _ in batch])
        t1_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s1_batch, a1_batch, r1_batch, t1_batch, s2_batch

    def clear(self):
        """
            initialize buffer  
        """
        self.buffer = []
        self.count = 0
