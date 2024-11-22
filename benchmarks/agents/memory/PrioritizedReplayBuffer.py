import collections
import math

import torch
from torch import FloatTensor, BoolTensor, IntTensor

from benchmarks.agents.memory.ReplayBufferInterface import ReplayBufferInterface


class PrioritizedReplayBuffer(ReplayBufferInterface):
    """
    Class implementing the prioritized experience replay buffer from:
    Tom Schaul. Prioritized experience replay. arXiv preprint arXiv:1511.05952, 2015.
    """

    def __init__(self, capacity=10000, max_priority=1e9):
        """
        Constructor.
        :param capacity: the number of experience the buffer can store
        """

        # Call parent constructor.
        super().__init__(capacity)

        # The weight associated with all the last experiences in the replay buffer.
        self.weights = collections.deque(maxlen=capacity)

        # The largest weight possible given to new transitions.
        self.max_priority = max_priority

        # The indices of the last sampled experiences.
        self.indices = collections.deque(maxlen=capacity)

    def append(self, experience):
        """
        Add a new experience to the buffer
        :param experience: the experience to add
        """
        super().append(experience)
        self.weights.append(self.max_priority)

    def sample(self, batch_size):
        """
        Sample a batch from the replay buffer.
        :param batch_size: the size of the batch to sample
        :return: observations, actions, rewards, done, next_observations
        where:
        - observations: the batch of observations
        - actions: the actions performed
        - rewards: the rewards received
        - done: whether the environment stop after performing the actions
        - next_observations: the observations received after performing the actions
        """

        # Sample a batch from the replay buffer.
        weights = torch.tensor(list(self.weights), dtype=torch.float)
        self.indices = torch.multinomial(weights, batch_size, replacement=False)
        obs, actions, rewards, done, next_obs = zip(*[self.buffer[idx] for idx in self.indices])

        # Convert the batch into a torch tensor stored on the proper device.
        return self.list_to_tensor(obs).to(self.device), \
            IntTensor(actions).to(self.device), \
            FloatTensor(rewards).to(self.device), \
            BoolTensor(done).to(self.device), \
            self.list_to_tensor(next_obs).to(self.device)

    def report(self, loss):
        """
        Report the loss associated with all the transitions of the previous batch.
        :param loss: the loss of all previous transitions
        """
        for i, idx in enumerate(self.indices):
            weight = loss[i].item()
            if math.isinf(weight) or math.isnan(weight):
                weight = self.max_priority
            self.weights[idx] = weight

    def clear(self):
        """
        Empty the replay buffer.
        """
        super().clear()
        self.weights.clear()
        self.indices.clear()
