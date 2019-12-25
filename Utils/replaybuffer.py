"""
    Replay Buffer for Deep Reinforcement Learning

"""

from collections import deque
import random
import numpy as np


class ReplayBuffer:
    def __init__(self, size_buffer, random_seed=None):
        if random_seed:
            random.seed(random_seed)
            np.random.seed(random_seed)
        self._size_bf = size_buffer
        self._length = 0
        self._buffer = deque()

    @property
    def buffer(self):
        return self._buffer

    def add(self, state, action, reward, state_next, done):
        exp = (state, action, reward, state_next, done)
        if self._length < self._size_bf:
            self._buffer.append(exp)
            self._length += 1
        else:
            self._buffer.popleft()
            self._buffer.append(exp)

    def add_batch(self, batch_s, batch_a, batch_r, batch_sn, batch_d):
        for i in range(len(batch_s)):
            self.add(batch_s[i], batch_a[i], batch_r[i], batch_sn[i], batch_d)

    def add_samples(self, samples):
        for s, a, r, sn, d in samples:
            self.add(s, a, r, sn, d)

    def __len__(self):
        return self._length

    def sample_batch(self, size_batch):

        if self._length < size_batch:
            batch = random.sample(self._buffer, self._length)
        else:
            batch = random.sample(self._buffer, size_batch)

        batch_s = np.array([d[0] for d in batch])
        batch_a = np.array([d[1] for d in batch])
        batch_r = np.array([d[2] for d in batch])
        batch_sn = np.array([d[3] for d in batch])
        batch_d = np.array([d[4] for d in batch])

        return batch_s, batch_a, batch_r, batch_sn, batch_d

    def clear(self):
        self._buffer.clear()
