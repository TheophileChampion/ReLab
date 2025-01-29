from typing import Optional

from relab import relab
from relab.cpp.agents.memory import FastReplayBuffer, CompressorType, Experience
from torch import Tensor

from relab.helpers.Typing import Config, Batch


class ReplayBuffer:
    """!
    @brief Python wrapper around replay buffer implemented in C++.

    @details
    The implementation is based on the following papers:

    <b>Prioritized experience replay</b>,
    published on arXiv, 2015.

    Authors:
    - Tom Schaul

    <b>Learning to predict by the methods of temporal differences</b>,
    published in Machine learning, 3:9–44, 1988.

    Authors:
    - Richard S. Sutton

    More precisely, the replay buffer supports multistep Q-learning and
    prioritization of experiences according to their associated loss.
    """

    def __init__(
        self,
        capacity : int = 10000,
        batch_size : int = 32,
        frame_skip : Optional[int] = None,
        stack_size : Optional[int] = None,
        screen_size : Optional[int] = None,
        args : Config = None
    ) -> None:
        """!
        Create a replay buffer.
        @param capacity: the number of experience the buffer can store
        @param batch_size: the size of the batch to sample
        @param frame_skip: the number of times each action is repeated in the environment, if None use the configuration
        @param stack_size: the number of stacked frame in each observation, if None use the configuration
        @param screen_size: the size of the images used by the agent to learn
        @param args: the prioritization and multistep arguments composed of:
            - initial_priority: the maximum experience priority given to new transitions
            - omega: the prioritization exponent
            - omega_is: the important sampling exponent
            - n_children: the maximum number of children each node of the priority-tree can have
            - n_steps: the number of steps for which rewards are accumulated in multistep Q-learning
            - gamma: the discount factor
        """

        stack_size = relab.config("stack_size") if stack_size is None else stack_size
        frame_skip = relab.config("frame_skip") if frame_skip is None else frame_skip
        screen_size = relab.config("screen_size") if screen_size is None else screen_size
        compressor_type = CompressorType.ZLIB if relab.config("compress_images") is True else CompressorType.RAW
        args = {} if args is None else args

        ## @var buffer
        # The C++ implementation of the replay buffer.
        self.buffer = FastReplayBuffer(
            capacity=capacity, batch_size=batch_size, frame_skip=frame_skip, stack_size=stack_size,
            screen_size=screen_size, type=compressor_type, args=args
        )

    def append(self, experience : Experience) -> None:
        """!
        Add a new experience to the buffer.
        @param experience: the experience to add
        """
        self.buffer.append(experience)

    def sample(self) -> Batch:
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

    def load(self, checkpoint_path : str = "", checkpoint_name : str = "") -> None:
        """!
        Load a replay buffer from the filesystem.
        @param checkpoint_path: the full checkpoint path from which the agent has been loaded
        @param checkpoint_name: the name of the checkpoint from which the replay buffer must be loaded ("" for default name)
        """
        self.buffer.load(checkpoint_path, checkpoint_name, relab.config("save_all_replay_buffers"))

    def save(self, checkpoint_path : str = "", checkpoint_name : str = "") -> None:
        """!
        Save the replay buffer on the filesystem.
        @param checkpoint_path: the full checkpoint path in which the agent has been saved
        @param checkpoint_name: the name of the checkpoint in which the replay buffer must be saved ("" for default name)
        """
        self.buffer.save(checkpoint_path, checkpoint_name, relab.config("save_all_replay_buffers"))

    def report(self, loss : Tensor) -> Tensor:
        """!
        Report the loss associated with all the transitions of the previous batch.
        @param loss: the loss of all previous transitions
        @return the new loss
        """
        return self.buffer.report(loss)

    def clear(self) -> None:
        """!
        Empty the replay buffer.
        """
        self.buffer.clear()

    def __len__(self) -> int:
        """!
        Retrieve the number of elements in the buffer.
        @return the number of elements contained in the replay buffer
        """
        return self.buffer.length()
