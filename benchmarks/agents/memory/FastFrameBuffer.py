from benchmarks.agents.memory.cpp import FrameBuffer


class FastFrameBuffer:
    """
    A buffer allowing for storage and retrieval of experience observations.
    """

    def __init__(self, capacity, frame_skip, n_steps, stack_size):
        """
        Create a frame buffer.
        :param capacity: the number of experiences the buffer can store
        :param frame_skip: the number of times each action is repeated in the environment
        :param n_steps: the number of steps for which rewards are accumulated in multistep Q-learning
        :param stack_size: the number of stacked frame in each observation
        """
        self.buffer = FrameBuffer(capacity, frame_skip, n_steps, stack_size)

    def append(self, experience):
        """
        Add the frames of the next experience to the buffer.
        :param experience: the experience whose frames must be added to the buffer
        """
        self.buffer.append(experience)

    def __getitem__(self, idx):
        """
        Retrieve the observations of the experience whose index is passed as parameters.
        :param idx: the index of the experience whose observations must be retrieved
        :return: the observations at time t and t + n_steps
        """
        return self.buffer.get(idx)

    def __len__(self):
        """
        Retrieve the number of experiences stored in the buffer.
        :return: the number of experiences stored in the buffer
        """
        return self.buffer.length()

    def clear(self):
        """
        Empty the frame buffer.
        """
        self.buffer.clear()

    def encode(self, frame):
        """
        Encode a frame to compress it.
        :param frame: the frame to encode
        :return: the encoded frame
        """
        return self.buffer.encode(frame)

    def decode(self, frame):
        """
        Decode a frame to decompress it.
        :param frame: the encoded frame to decode
        :return: the decoded frame
        """
        return self.buffer.decode(frame)
