import numpy as np
from torch import FloatTensor, BoolTensor, IntTensor

from benchmarks.agents.memory.ReplayBufferInterface import ReplayBufferInterface


class ReplayBuffer(ReplayBufferInterface):
    """
    Class implementing the experience replay buffer.
    """

    def __init__(self, capacity=10000):
        """
        Constructor.
        :param capacity: the number of experience the buffer can store
        """
        super().__init__(capacity)

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
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        obs, actions, rewards, done, next_obs = zip(*[self.buffer[idx] for idx in indices])

        # Convert the batch into a torch tensor stored on the proper device.
        return self.list_to_tensor(obs).to(self.device), \
            IntTensor(actions).to(self.device), \
            FloatTensor(rewards).to(self.device), \
            BoolTensor(done).to(self.device), \
            self.list_to_tensor(next_obs).to(self.device)
