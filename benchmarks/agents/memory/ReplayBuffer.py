import numpy as np

from benchmarks import benchmarks

import collections
from torch import cat, unsqueeze
import benchmarks
import math
import torch

from benchmarks.agents.memory.DataBuffer import DataBuffer
from benchmarks.agents.memory.FrameBuffer import FrameBuffer


#
# Class storing an experience.
#
Experience = collections.namedtuple("Experience", field_names=["obs", "action", "reward", "done", "next_obs"])


class ReplayBuffer:
    """
    Class implementing the prioritized experience replay buffer from:
    Tom Schaul. Prioritized experience replay. arXiv preprint arXiv:1511.05952, 2015.
    """

    def __init__(self, capacity=10000, batch_size=32, frame_skip=None, stack_size=None, p_args=None, m_args=None):
        """
        Create a replay buffer.
        :param capacity: the number of experience the buffer can store
        :param batch_size: the size of the batch to sample
        :param frame_skip: the number of times each action is repeated in the environment, if None use the configuration
        :param stack_size: the number of stacked frame in each observation, if None use the configuration
        :param p_args: the prioritization arguments (None for no prioritization) composed of:
            - initial_priority: the maximum experience priority given to new transitions
            - omega: the prioritization exponent
            - omega_is: the important sampling exponent
            - n_children: the maximum number of children each node of the priority-tree can have
        :param m_args: the multistep arguments (None for no multistep) composed of:
            - n_steps: the number of steps for which rewards are accumulated in multistep Q-learning
            - gamma: the discount factor
        """

        # Keep in mind whether the replay buffer is prioritized.
        self.prioritized = (p_args is not None)

        # Default values of the prioritization, multistep, worker arguments.
        default_p_args = {"initial_priority": 1.0, "omega": 1.0, "omega_is": 1.0, "n_children": 10}
        default_m_args = {"n_steps": 1, "gamma": 0.99}

        # Overwrite default values if prioritization or multistep arguments are provided.
        p_args = default_p_args if p_args is None else default_p_args | p_args
        m_args = default_m_args if m_args is None else default_m_args | m_args

        # Store the buffer parameters, and the device on which computation is performed.
        self.capacity = capacity
        self.batch_size = batch_size
        self.stack_size = benchmarks.config("stack_size") if stack_size is None else stack_size
        self.frame_skip = benchmarks.config("frame_skip") if frame_skip is None else frame_skip
        self.gamma = m_args["gamma"]
        self.n_steps = m_args["n_steps"]
        self.initial_priority = p_args["initial_priority"]
        self.n_children = p_args["n_children"]
        self.omega = p_args["omega"]
        self.omega_is = p_args["omega_is"]
        self.device = benchmarks.device()

        # The buffer storing the frames of all experiences.
        self.observations = FrameBuffer(self.capacity, self.frame_skip, self.n_steps, self.stack_size)

        # The buffer storing the data (i.e., actions, rewards, dones and priorities) of all experiences.
        self.data = DataBuffer(self.capacity, self.n_steps, self.gamma, self.initial_priority, self.n_children)

        # The indices of the last sampled experiences.
        self.indices = []

    def append(self, experience):
        """
        Add a new experience to the buffer
        :param experience: the experience to add
        """
        self.observations.append(experience)
        self.data.append(experience)

    def sample(self):
        """
        Sample a batch from the replay buffer.
        :return: observations, actions, rewards, done, next_observations
        where:
        - observations: the batch of observations
        - actions: the actions performed
        - rewards: the rewards received
        - done: whether the environment stop after performing the actions
        - next_observations: the observations received after performing the actions
        """

        # Sample a batch from the replay buffer.
        if self.prioritized:
            self.indices = self.data.priorities.sample_indices(self.batch_size)
        else:
            self.indices = np.random.choice(len(self), self.batch_size)

        obs, actions, rewards, done, next_obs = self.get_experiences(self.indices)

        # Convert the batch into a torch tensor stored on the proper device.
        return obs.to(self.device), actions, rewards, done, next_obs.to(self.device)

    def report(self, loss):
        """
        Report the loss associated with all the transitions of the previous batch.
        :param loss: the loss of all previous transitions
        """

        # If the buffer is not prioritized, don't update the priorities.
        if not self.prioritized:
            return loss

        # Add a small positive constant to avoid zero probabilities.
        loss += 1e-5

        # Raise the loss to the power of the prioritization exponent.
        if self.omega != 1.0:
            loss = loss.pow(self.omega)

        # If the buffer is prioritized, update the priorities.
        sum_priorities = self.data.priorities.sum()
        priorities = torch.zeros([self.batch_size])
        for i, idx in enumerate(self.indices):
            priority = loss[i].item()
            if math.isinf(priority) or math.isnan(priority):
                priority = self.data.priorities.max()
            priorities[i] = self.data.priorities[idx]
            self.data.priorities[idx] = priority

        # Update the priorities and compute the importance sampling weights.
        weights = len(self) * priorities.to(self.device) / sum_priorities
        weights = torch.pow(weights, -self.omega_is)
        return loss * weights / weights.max()

    def get_experiences(self, indices):
        """
        Retrieve the experiences whose indices are passed as parameters.
        :param indices: the experience indices
        :return: the experiences
        """
        obs, next_obs = zip(*[self.observations[idx] for idx in indices])
        actions, rewards, done = self.data[indices]
        return self.list_to_tensor(obs), actions, rewards, done, self.list_to_tensor(next_obs)

    def __len__(self):
        """
        Retrieve the number of elements in the buffer.
        :return: the number of elements contained in the replay buffer
        """
        return len(self.observations)

    def clear(self):
        """
        Empty the replay buffer.
        """
        self.observations.clear()
        self.data.clear()
        self.indices = []

    @staticmethod
    def list_to_tensor(tensor_list):
        """
        Transform a list of n dimensional tensors into a tensor with n+1 dimensions.
        :param tensor_list: the list of tensors
        :return: the output tensor
        """
        return cat([unsqueeze(tensor, 0) for tensor in tensor_list])
