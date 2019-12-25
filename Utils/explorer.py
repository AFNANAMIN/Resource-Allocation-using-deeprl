import numpy as np


class Explorer:
    def __init__(self, epsilon_begin, epsilon_end, steps):
        self._ep_b = epsilon_begin
        self._ep_e = epsilon_end
        self._ep = epsilon_begin
        self._counter_random = 0
        self._steps = steps
        self._st_counter = 0

    @property
    def counter(self):
        return self._counter_random

    @property
    def epsilon(self):
        return self._ep

    def get_action(self, q_value):
        shape = np.array(q_value).shape
        assert len(shape) == 1, 'q_value shape error {0}'.format(shape)

        action = np.zeros(shape[0])

        if np.random.random() <= self._ep:
            action_index = np.random.randint(low=0, high=shape[0])
            self._counter_random += 1
        else:
            action_index = np.argmax(q_value)

        action[action_index] = 1

        self._st_counter += 1

        return action

    def decay(self):
        if self._ep > self._ep_e:
            self._ep -= (self._ep_b - self._ep_e) / self._steps

    def get_pure_action(self, q_value):
        shape = np.array(q_value).shape
        assert len(shape) == 1, 'q_value shape error {0}'.format(shape)

        action = np.zeros(shape[0])
        action[np.argmax(q_value)] = 1

        return action
