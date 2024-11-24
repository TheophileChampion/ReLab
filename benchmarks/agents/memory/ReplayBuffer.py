import collections
import math

import numpy as np
from torch import FloatTensor, BoolTensor, IntTensor, cat, unsqueeze
import torch

import benchmarks


#
# Class storing an experience.
#
Experience = collections.namedtuple("Experience", field_names=["obs", "action", "reward", "done", "next_obs"])


class ReplayBuffer:
    """
    Class implementing a prioritized replay buffer [1] with support for multistep Q-learning [2] from:

    [1] Tom Schaul. Prioritized experience replay. arXiv preprint arXiv:1511.05952, 2015.
    [2] Richard S Sutton. Learning to predict by the methods of temporal differences. Machine learning, 3:9–44, 1988.
    """

    def __init__(self, capacity=10000, max_priority=None, n_steps=1, gamma=0.99):
        """
        Constructor.
        :param capacity: the number of experience the buffer can store
        :param max_priority: the maximum experience priority given to new transitions, None for no prioritization
        :param n_steps: the number of steps for which rewards are accumulated in multistep Q-learning
        :param gamma: the discount factor
        """

        # The experience replay buffer and the device on which to store experiences.
        self.buffer = collections.deque(maxlen=capacity)
        self.device = benchmarks.device()

        # Store the multistep parameters.
        self.gamma = gamma
        self.n_steps = n_steps
        self.last_experiences = collections.deque(maxlen=n_steps)

        # The largest weight possible given to new transitions.
        self.max_priority = max_priority

        # The weight associated with all the last experiences in the replay buffer.
        self.weights = [] if max_priority is None else collections.deque(maxlen=capacity)

        # The indices of the last sampled experiences.
        self.indices = []

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
        if self.max_priority is None:
            self.indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        else:
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

        # If no prioritization is required, simply return.
        if self.max_priority is None:
            return

        # Update the prioritization weights.
        for i, idx in enumerate(self.indices):
            weight = loss[i].item()
            if math.isinf(weight) or math.isnan(weight):
                weight = self.max_priority
            self.weights[idx] = weight

    @staticmethod
    def list_to_tensor(tensor_list):
        """
        Transform a list of n dimensional tensors into a tensor with n+1 dimensions.
        :param tensor_list: the list of tensors
        :return: the output tensor
        """
        return cat([unsqueeze(tensor, 0) for tensor in tensor_list])

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

        # Update the cumulated return of (past) experiences.
        if self.n_steps > 1:

            # Update the returns of last experiences.
            for i in range(len(self.last_experiences)):
                new_reward = self.last_experiences[i].reward + math.pow(self.gamma, i + 1) * experience.reward
                self.last_experiences[i] = self.last_experiences[i]._replace(reward=new_reward)

            # Add the current experience to the queue of last experiences.
            self.last_experiences.appendleft(experience)

            # Check if no experience should be added to the replay buffer, simply return.
            if len(self.last_experiences) < self.n_steps:
                return

            # Otherwise pop the experience from the queue of (past) experiences and update its next observation.
            experience = self.last_experiences.pop()
            experience = experience._replace(next_obs=self.last_experiences[0].next_obs)

        # Add the experience to the replay buffer and initialize its weight (if needed).
        self.buffer.append(experience)
        if self.max_priority is not None:
            self.weights.append(self.max_priority)

    def clear(self):
        """
        Empty the replay buffer.
        """
        self.buffer.clear()
        self.weights.clear()
        self.indices.clear()
        self.last_experiences.clear()
