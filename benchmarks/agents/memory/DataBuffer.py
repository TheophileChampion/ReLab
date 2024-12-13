from benchmarks import benchmarks

import collections
import benchmarks
import math
import torch

from benchmarks.agents.memory.PriorityTree import PriorityTree


class DataBuffer:
    """
    A buffer allowing for storage and retrieval of experience datum (i.e., action, reward, done, and priority).
    """

    def __init__(self, capacity, n_steps, gamma, initial_priority, n_children):
        """
        Create a data buffer.
        :param capacity: the number of experiences the buffer can store
        :param n_steps: the number of steps for which rewards are accumulated in multistep Q-learning
        :param gamma: the discount factor
        :param initial_priority: the initial priority given to first elements
        :param n_children: the number of children each node has
        """

        # Store the data buffer's parameters.
        self.capacity = capacity
        self.n_steps = n_steps
        self.gamma = gamma

        # Queues keeping track of past actions, cumulated rewards, and dones.
        self.past_actions = collections.deque(maxlen=n_steps)
        self.past_rewards = collections.deque(maxlen=n_steps)
        self.past_dones = collections.deque(maxlen=n_steps)

        # Torch tensors storing all the buffer's data.
        self.device = benchmarks.device()
        self.actions = torch.zeros([capacity], dtype=torch.int).to(self.device)
        self.rewards = torch.zeros([capacity], dtype=torch.float).to(self.device)
        self.dones = torch.zeros([capacity], dtype=torch.bool).to(self.device)

        # The priorities associated with all experiences in the replay buffer.
        self.priorities = PriorityTree(capacity=capacity, initial_priority=initial_priority, n_children=n_children)

        # The index of the next datum to add in the buffer.
        self.current_id = 0

    def append(self, experience):
        """
        Add the datum of the next experience to the buffer.
        :param experience: the experience whose datum must be added to the buffer
        """

        # Update the returns of last experiences.
        for i in range(len(self.past_rewards)):
            self.past_rewards[i] += math.pow(self.gamma, i + 1) * experience.reward

        # Add the reward, action and done to their respective queues.
        self.past_rewards.appendleft(experience.reward)
        self.past_actions.appendleft(experience.action)
        self.past_dones.appendleft(experience.done)

        # Add new data to the buffer.
        if experience.done is True:

            # If the current episode has ended, keep track of all valid data.
            while len(self.past_rewards) != 0:
                self.add_datum(self.past_actions.pop(), self.past_rewards.pop(), self.past_dones[0])

            # Then, clear the queues of past reward, actions, and dones.
            self.past_rewards.clear()
            self.past_actions.clear()
            self.past_dones.clear()

        elif len(self.past_rewards) == self.n_steps:

            # If the current episode has not ended, but the queues are full, then keep track of next valid datum.
            self.add_datum(self.past_actions.pop(), self.past_rewards.pop(), self.past_dones[0])

    def __getitem__(self, indices):
        """
        Retrieve the data of the experience whose indices are passed as parameters.
        :param indices: the indices of the experiences whose data must be retrieved
        :return: the data (i.e., action at time t, n-steps return at time t, and done at time t + n_steps)
        """
        if self.current_id >= self.capacity:
            indices = [(index + self.current_id) % self.capacity for index in indices]
        return self.actions[indices], self.rewards[indices], self.dones[indices]

    def __len__(self):
        """
        Retrieve the number of experiences stored in the buffer.
        :return: the number of experiences stored in the buffer
        """
        return min(self.current_id, self.capacity)

    def clear(self):
        """
        Empty the data buffer.
        """
        self.past_actions.clear()
        self.past_rewards.clear()
        self.past_dones.clear()
        self.actions = torch.zeros([self.capacity])
        self.rewards = torch.zeros([self.capacity])
        self.dones = torch.zeros([self.capacity])
        self.priorities.clear()
        self.current_id = 0

    def add_datum(self, action, reward, done):
        """
        Add a datum to the buffer.
        :param action: the action at time t
        :param reward: the n-steps reward
        :param done: the done at time t + n_steps
        """
        index = self.current_id % self.capacity
        self.actions[index] = action
        self.rewards[index] = reward
        self.dones[index] = done
        self.priorities.append(self.priorities.max())
        self.current_id += 1
