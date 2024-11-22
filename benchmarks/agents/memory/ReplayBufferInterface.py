import abc
from abc import ABC
import collections
from torch import cat, unsqueeze

import benchmarks


#
# Class storing an experience.
#
Experience = collections.namedtuple("Experience", field_names=["obs", "action", "reward", "done", "next_obs"])


class ReplayBufferInterface(ABC):
    """
    The interface that all replay buffers must implement.
    """

    def __init__(self, capacity=10000):
        """
        Constructor.
        :param capacity: the number of experience the buffer can store
        """
        self.buffer = collections.deque(maxlen=capacity)
        self.device = benchmarks.device()

    @abc.abstractmethod
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
        ...

    def report(self, loss):
        """
        Report the loss associated with all the transitions of the previous batch.
        :param loss: the loss of all previous transitions
        """
        ...

    def __len__(self):
        """
        Retrieve the number of elements in the buffer.
        :return: the number of elements contained in the replay buffer
        """
        return len(self.buffer)

    def append(self, experience):
        """
        Add a new experience to the buffer
        :param experience: the experience to add
        """
        self.buffer.append(experience)

    @staticmethod
    def list_to_tensor(tensor_list):
        """
        Transform a list of n dimensional tensors into a tensor with n+1 dimensions.
        :param tensor_list: the list of tensors
        :return: the output tensor
        """
        return cat([unsqueeze(tensor, 0) for tensor in tensor_list])

    def clear(self):
        """
        Empty the replay buffer.
        """
        self.buffer.clear()
