import logging

from relab import relab
from relab.agents.memory.cpp import FastReplayBuffer, CompressorType
from relab.helpers.FileSystem import FileSystem
import os
from os.path import join, isfile


class ReplayBuffer:
    """!
    Class implementing a replay buffer with support for prioritization [1] and multistep Q-learning [2] from:

    [1] Tom Schaul. Prioritized experience replay. arXiv preprint arXiv:1511.05952, 2015.
    [2] Richard S Sutton. Learning to predict by the methods of temporal differences. Machine learning, 3:9–44, 1988.
    """

    def __init__(self, capacity=10000, batch_size=32, frame_skip=None, stack_size=None, screen_size=None, p_args=None, m_args=None):
        """!
        Create a replay buffer.
        @param capacity: the number of experience the buffer can store
        @param batch_size: the size of the batch to sample
        @param frame_skip: the number of times each action is repeated in the environment, if None use the configuration
        @param stack_size: the number of stacked frame in each observation, if None use the configuration
        @param screen_size: the size of the images used by the agent to learn
        @param p_args: the prioritization arguments (None for no prioritization) composed of:
            - initial_priority: the maximum experience priority given to new transitions
            - omega: the prioritization exponent
            - omega_is: the important sampling exponent
            - n_children: the maximum number of children each node of the priority-tree can have
        @param m_args: the multistep arguments (None for no multistep) composed of:
            - n_steps: the number of steps for which rewards are accumulated in multistep Q-learning
            - gamma: the discount factor
        """

        stack_size = relab.config("stack_size") if stack_size is None else stack_size
        frame_skip = relab.config("frame_skip") if frame_skip is None else frame_skip
        screen_size = relab.config("screen_size") if screen_size is None else screen_size
        compressor_type = CompressorType.ZLIB if relab.config("compress_images") is True else CompressorType.RAW
        p_args = {} if p_args is None else p_args
        m_args = {} if m_args is None else m_args

        ## @var buffer
        # The C++ implementation of the replay buffer.
        self.buffer = FastReplayBuffer(
            capacity=capacity, batch_size=batch_size, frame_skip=frame_skip, stack_size=stack_size,
            screen_size=screen_size, type=compressor_type, p_args=p_args, m_args=m_args
        )

    def append(self, experience):
        """!
        Add a new experience to the buffer.
        @param experience: the experience to add
        """
        self.buffer.append(experience)

    def sample(self):
        """!
        Sample a batch from the replay buffer.
        @return observations, actions, rewards, done, next_observations
        where:
        - observations: the batch of observations
        - actions: the actions performed
        - rewards: the rewards received
        - done: whether the environment stop after performing the actions
        - next_observations: the observations received after performing the actions
        """
        return self.buffer.sample()

    def load(self, checkpoint_path=None, checkpoint_name=None):
        """!
        Load a replay buffer from the filesystem.
        @param checkpoint_path: the full checkpoint path from which the agent has been loaded
        @param checkpoint_name: the name of the checkpoint from which the replay buffer must be loaded (None for default name)
        """

        # TODO move this to c++ code?
        # Retrieve the full replay buffer checkpoint path.
        if checkpoint_name is None and relab.config("save_all_replay_buffers") is False:
            checkpoint_name = "buffer.pt"
        if checkpoint_name is None:
            buffer_checkpoint_path = checkpoint_path.replace("model_", "buffer_")
        else:
            buffer_checkpoint_path = join(os.environ["CHECKPOINT_DIRECTORY"], checkpoint_name)

        # Check that the replay buffer checkpoint exist.
        if not isfile(buffer_checkpoint_path):
            logging.info(f"Could not load the replay buffer from: {buffer_checkpoint_path}")
            return

        # Load the replay buffer from the filesystem.
        self.buffer.load(buffer_checkpoint_path)

    def save(self, checkpoint_path=None, checkpoint_name=None):
        """!
        Save the replay buffer on the filesystem.
        @param checkpoint_path: the full checkpoint path in which the agent has been saved
        @param checkpoint_name: the name of the checkpoint in which the replay buffer must be saved (None for default name)
        """

        # TODO move this to c++ code?
        # Create the replay buffer checkpoint directory and file, if they do not exist.
        if checkpoint_name is None and relab.config("save_all_replay_buffers") is False:
            checkpoint_name = "buffer.pt"
        if checkpoint_name is None:
            buffer_checkpoint_path = checkpoint_path.replace("model_", "buffer_")
        else:
            buffer_checkpoint_path = join(os.environ["CHECKPOINT_DIRECTORY"], checkpoint_name)
        FileSystem.create_directory_and_file(buffer_checkpoint_path)

        # Save the replay buffer on the filesystem.
        self.buffer.save(buffer_checkpoint_path)

    def report(self, loss):
        """!
        Report the loss associated with all the transitions of the previous batch.
        @param loss: the loss of all previous transitions
        @return the new loss
        """
        return self.buffer.report(loss)

    def clear(self):
        """!
        Empty the replay buffer.
        """
        self.buffer.clear()

    def __len__(self):
        """!
        Retrieve the number of elements in the buffer.
        @return the number of elements contained in the replay buffer
        """
        return self.buffer.length()
