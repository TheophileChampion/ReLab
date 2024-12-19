from benchmarks.agents.memory.cpp import DataBuffer


class FastDataBuffer:
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
        self.buffer = DataBuffer(capacity, n_steps, gamma, initial_priority, n_children)

    def append(self, experience):
        """
        Add the datum of the next experience to the buffer.
        :param experience: the experience whose datum must be added to the buffer
        """
        self.buffer.append(experience)

    def __getitem__(self, indices):
        """
        Retrieve the data of the experience whose indices are passed as parameters.
        :param indices: the indices of the experiences whose data must be retrieved
        :return: the data (i.e., action at time t, n-steps return at time t, and done at time t + n_steps)
        """
        return self.buffer.get(indices)

    def __len__(self):
        """
        Retrieve the number of experiences stored in the buffer.
        :return: the number of experiences stored in the buffer
        """
        return self.buffer.length()

    def clear(self):
        """
        Empty the data buffer.
        """
        return self.buffer.clear()
