from torchvision.io import encode_png, decode_png

import collections
import torch

from benchmarks.agents.memory.CircularList import CircularList

# Class storing references to the first frame of observations at time t and t + n_steps.
ObsReferences = collections.namedtuple("ObsReferences", field_names=["frame_t", "frame_tn"])

# Class storing a frame.
Frame = collections.namedtuple("Frame", field_names=["index", "data"])


class FrameBuffer:
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

        # Store the frame buffer's parameters.
        self.frame_skip = frame_skip
        self.stack_size = stack_size
        self.capacity = capacity
        self.n_steps = n_steps

        # A circular list storing all the buffer's frames.
        self.frames = CircularList(capacity)

        # A list storing the observation references of each experience.
        self.references = [ObsReferences(None, None)] * capacity
        self.current_ref = 0

        # A queue storing the frames of recent observations (for multistep Q-learning).
        self.past_obs_frames = collections.deque(maxlen=n_steps + 1)

        # A boolean keeping track of whether the next experience is the beginning of a new episode.
        self.new_episode = True

    def append(self, experience):
        """
        Add the frames of the next experience to the buffer.
        :param experience: the experience whose frames must be added to the buffer
        """

        # If the buffer is full, remove the oldest observation frames from the buffer.
        if len(self) == self.capacity:
            while self.frames[0] != self.references[self.first_reference() % self.capacity].frame_t:
                self.frames.pop()

        # Add the frames of the observation at time t, if needed.
        if self.new_episode is True:
            for i in range(self.stack_size):
                self.add_frame(self.encode(experience.obs[i, :, :]))
            self.past_obs_frames.append(self.frames[-self.stack_size])

        # Add the frames of the observation at time t + 1.
        n = min(self.frame_skip, self.stack_size)
        for i in reversed(range(1, n + 1)):
            self.add_frame(self.encode(experience.next_obs[-i, :, :]))
        self.past_obs_frames.append(self.frames[-self.stack_size])

        # Update the observation references.
        if experience.done is True:

            # If the current episode has ended, keep track of all valid references.
            while len(self.past_obs_frames) != 1:
                self.add_reference(t_0=0, t_1=-1)
                self.past_obs_frames.popleft()

            # Then, clear the queue of past observation frames.
            self.past_obs_frames.clear()

        elif len(self.past_obs_frames) == self.n_steps + 1:

            # If the current episode has not ended, but the queue of past observation frame is full,
            # then keep track of next valid reference (before it is discarded in the next call to append).
            self.add_reference(t_0=0, t_1=self.n_steps)

        # Keep track of whether the next experience is the beginning of a new episode.
        self.new_episode = experience.done

    def __getitem__(self, idx):
        """
        Retrieve the observations of the experience whose index is passed as parameters.
        :param idx: the index of the experience whose observations must be retrieved
        :return: the observations at time t and t + n_steps
        """

        # Retrieve the index of first frame for the requested observations.
        reference = self.references[(self.first_reference() + idx) % self.capacity]
        idx_t = reference.frame_t.index - self.frames[0].index
        idx_tn = reference.frame_tn.index - self.frames[0].index

        # Retrieve the requested observations.
        return self.get_observation(idx_t), self.get_observation(idx_tn)

    def __len__(self):
        """
        Retrieve the number of experiences stored in the buffer.
        :return: the number of experiences stored in the buffer
        """
        return min(self.current_ref, self.capacity)

    def clear(self):
        """
        Empty the frame buffer.
        """
        self.frames.clear()
        self.references = [None] * self.capacity
        self.past_obs_frames.clear()
        self.new_episode = True
        self.current_ref = 0

    def add_frame(self, data):
        """
        Add a frame to the buffer.
        :param data: the frame's data
        """
        self.frames.append(Frame(index=self.frames.current_index, data=data))

    def add_reference(self, t_0, t_1):
        """
        Add an observation references to the buffer.
        :param t_0: the index of the first frame in the queue of past observation frames
        :param t_1: the index of the second frame in the queue of past observation frames
        """
        self.references[self.current_ref % self.capacity] = \
            ObsReferences(frame_t=self.past_obs_frames[t_0], frame_tn=self.past_obs_frames[t_1])
        self.current_ref += 1

    def get_observation(self, idx):
        """
        Retrieve an observation from the buffer.
        :param idx: the index of the first frame of the observation to retrieve
        """
        frames = []
        for i in range(self.stack_size):
            frames.append(self.decode(self.frames[idx + i].data))
        return torch.concat(frames)

    def first_reference(self):
        """
        Retrieve the index of the first reference of the buffer.
        :return: the index
        """
        return 0 if self.current_ref < self.capacity else self.current_ref

    @staticmethod
    def encode(frame):
        """
        Encode a frame using PNG.
        :param frame: the frame to encode
        :return: the encoded frame
        """
        return encode_png((frame * 255).type(torch.uint8).unsqueeze(dim=0))

    @staticmethod
    def decode(frame):
        """
        Decode a PNG frame.
        :param frame: the PNG frame to decode
        :return: the decoded frame
        """
        return decode_png(frame).type(torch.float32) / 255
