import abc
import errno
import logging
import math
import os
import re
import time
from abc import ABC
from collections import deque
from enum import IntEnum
from functools import partial
from os.path import exists, isdir, isfile, join
from typing import Any, Callable, Dict, Optional, SupportsFloat

import imageio
import numpy as np
import psutil
import relab
import torch
from gymnasium import Env
from PIL import Image
from relab.agents.memory.ReplayBuffer import ReplayBuffer
from relab.helpers.FileSystem import FileSystem
from relab.helpers.Serialization import safe_load
from relab.helpers.Typing import (
    ActionType,
    AttributeNames,
    Checkpoint,
    Config,
    ObservationType,
)
from torch.utils.tensorboard import SummaryWriter


class ReplayType(IntEnum):
    """!
    The type of replay buffer supported by the agents.
    """

    # @var DEFAULT
    # Standard replay buffer with uniform sampling.
    DEFAULT = 0

    # @var PRIORITIZED
    # Prioritized experience replay buffer that samples transitions based on
    # their associated loss values.
    PRIORITIZED = 1

    # @var MULTISTEP
    # Replay buffer that stores n-step transitions for multistep Q-learning.
    MULTISTEP = 2

    # @var MULTISTEP_PRIORITIZED
    # Combination of prioritized experience replay and n-step transitions.
    MULTISTEP_PRIORITIZED = 3


class AgentInterface(ABC):
    """!
    @brief The interface that all agents must implement.
    """

    def __init__(
        self,
        get_buffer: Optional[Callable] = None,
        n_actions: int = 18,
        training: bool = True,
    ) -> None:
        """!
        Create an agent.
        @param get_buffer: a function returning the replay buffer used to store the agent's experiences
        @param n_actions: the number of actions available to the agent
        @param training: True if the agent is being training, False otherwise
        """

        # @var n_actions
        # Number of possible actions available to the agent.
        self.n_actions = n_actions

        # @var training
        # Flag indicating whether the agent is in training mode.
        self.training = training

        # @var device
        # The device (CPU/GPU) used for training computations.
        self.device = relab.device()

        # @var max_queue_len
        # Maximum length for metric tracking queues (e.g., rewards, losses).
        self.max_queue_len = 100

        # @var current_step
        # Counter tracking the number of training steps performed.
        self.current_step = 0

        # @var vfe_losses
        # Queue containing the last variational free energy loss values.
        self.vfe_losses = deque(maxlen=self.max_queue_len)

        # @var betas
        # Queue containing the last beta values for variational inference.
        self.betas = deque(maxlen=self.max_queue_len)

        # @var log_likelihoods
        # Queue containing the last log-likelihood values.
        self.log_likelihoods = deque(maxlen=self.max_queue_len)

        # @var kl_divergences
        # Queue containing the last KL-divergence values.
        self.kl_divergences = deque(maxlen=self.max_queue_len)

        # @var efe_losses
        # Queue containing the last EFE loss values.
        self.efe_losses = deque(maxlen=self.max_queue_len)

        # @var process
        # Object representing the current process, used to track memory usage.
        self.process = psutil.Process()

        # @var residential_memory
        # Queue tracking residential memory usage over time.
        self.residential_mem = deque(maxlen=self.max_queue_len)

        # @var episodic_rewards
        # Queue containing the last episodic reward values.
        self.episodic_rewards = deque(maxlen=self.max_queue_len)

        # @var current_episodic_reward
        # Accumulator for the current episode's reward.
        self.curr_ep_reward = 0

        # @var time_elapsed
        # Queue containing the time elapsed between consecutive training
        # iterations.
        self.time_elapsed = deque(maxlen=self.max_queue_len)

        # @var last_time
        # Timestamp of the last training iteration.
        self.last_time = None

        # @var episode_lengths
        # Queue containing the lengths of recent episodes.
        self.episode_lengths = deque(maxlen=self.max_queue_len)

        # @var current_episode_length
        # Counter for the current episode's length.
        self.curr_ep_length = 0

        # @var checkpoint_dir
        # Directory in which to save the checkpoints.
        self.checkpoint_dir = os.environ["CHECKPOINT_DIRECTORY"]

        # @var tensorboard_dir
        # TensorBoard directory in which to log training metrics.
        self.tensorboard_dir = os.environ["TENSORBOARD_DIRECTORY"]

        # @var writer
        # TensorBoard summary writer for logging training metrics.
        self.writer = SummaryWriter(self.tensorboard_dir) if training else None

        # @var get_buffer
        # A function that creates a new instance of the replay buffer.
        self.get_buffer = get_buffer

        # @var buffer
        # Experience replay buffer for storing transitions.
        self.buffer = None if get_buffer is None or training is False else get_buffer()

    @abc.abstractmethod
    def step(self, obs: ObservationType) -> ActionType:
        """!
        Select the next action to perform in the environment.
        @param obs: the observation available to make the decision
        @return the next action to perform
        """
        ...

    @abc.abstractmethod
    def train(self, env: Env) -> None:
        """!
        Train the agent in the gym environment passed as parameters
        @param env: the gym environment
        """
        ...

    def load(
        self,
        checkpoint_name: str = "",
        buffer_checkpoint_name: str = "",
        attr_names: Optional[AttributeNames] = None,
    ) -> Checkpoint:
        """!
        Load an agent from the filesystem.
        @param checkpoint_name: the name of the agent checkpoint to load
        @param buffer_checkpoint_name: the name of the replay buffer checkpoint to load (None for default name)
        @param attr_names: a list of attribute names to load from the checkpoint (load all attributes by default)
        @return the loaded checkpoint object
        """

        # Retrieve the full agent checkpoint path.
        if checkpoint_name == "":
            checkpoint_path = self.get_latest_checkpoint()
        else:
            checkpoint_path = join(self.checkpoint_dir, checkpoint_name)

        # Check if the checkpoint can be loaded.
        if checkpoint_path is None:
            logging.info("Could not load the agent from the file system.")
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), "checkpoint.pt"
            )

        # Load the checkpoint from the file system.
        logging.info(f"Loading agent from: {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        # Retrieve the full list of attribute names.
        attr_names = [] if attr_names is None else list(attr_names)
        attr_names += list(AgentInterface.as_dict(self).keys())

        # Load the class attributes from the checkpoint.
        exclude_names = [
            "optimizer",
            "encoder",
            "decoder",
            "transition_net",
            "value_net",
            "target_net",
        ]
        for name in attr_names:
            if name not in exclude_names:
                setattr(self, name, safe_load(checkpoint, name))

        # Initialize the last time to minus one to signal that the buffer was reloaded.
        self.last_time = -1

        # Load the replay buffer from the checkpoint.
        if self.get_buffer is not None and self.training is True:
            self.buffer = self.get_buffer()
        else:
            self.buffer = None
        self.buffer.load(checkpoint_path, buffer_checkpoint_name)
        return checkpoint

    def as_dict(self) -> Config:
        """!
        Convert the agent into a dictionary that can be saved on the filesystem.
        @return the dictionary
        """
        return {
            "training": self.training,
            "current_step": self.current_step,
            "max_queue_len": self.max_queue_len,
            "curr_ep_reward": self.curr_ep_reward,
            "curr_ep_length": self.curr_ep_length,
            "vfe_losses": self.vfe_losses,
            "betas": self.betas,
            "log_likelihoods": self.log_likelihoods,
            "kl_divergences": self.kl_divergences,
            "efe_losses": self.efe_losses,
            "residential_mem": self.residential_mem,
            "episodic_rewards": self.episodic_rewards,
            "time_elapsed": self.time_elapsed,
            "episode_lengths": self.episode_lengths,
            "checkpoint_dir": self.checkpoint_dir,
            "tensorboard_dir": self.tensorboard_dir,
        }

    def save(
        self,
        checkpoint_name: str,
        buffer_checkpoint_name: str = "",
        agent_conf: Optional[Config] = None,
    ) -> None:
        """!
        Save the agent on the filesystem.
        @param checkpoint_name: the name of the checkpoint in which to save the agent
        @param buffer_checkpoint_name: the name of the checkpoint to save the replay buffer (None for default name)
        @param agent_conf: a dictionary representing the agent's attributes to be saved (for internal use only)
        """

        # Create the checkpoint directory and file, if they do not exist.
        checkpoint_path = join(self.checkpoint_dir, checkpoint_name)
        FileSystem.create_directory_and_file(checkpoint_path)

        # Save the model.
        logging.info(f"Saving agent to the following file: {checkpoint_path}")
        torch.save(agent_conf | AgentInterface.as_dict(self), checkpoint_path)

        # Save the replay buffer.
        if self.buffer is not None:
            self.buffer.save(checkpoint_path, buffer_checkpoint_name)

    def demo(self, env: Env, gif_name: str, max_frames: int = 10000) -> None:
        """!
        Demonstrate the agent policy in the gym environment passed as parameters
        @param env: the gym environment
        @param gif_name: the name of the GIF file in which to save the demo
        @param max_frames: the maximum number of frames to include in the GIF file
        """

        # Reset the environment.
        obs, _ = env.reset()

        # Record the agent policy.
        frames = []
        for t in range(max_frames):

            # Record the frame associated to the current environment state.
            frames.append(Image.fromarray(env.render().numpy()))

            # Execute an action in the environment.
            action = self.step(obs.to(self.device))
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Stop recording if the episode ends.
            if done:
                break

        # Close the environment.
        env.close()

        # Create a GIF of the recorded frames.
        gif_path = join(os.environ["DEMO_DIRECTORY"], gif_name)
        FileSystem.create_directory_and_file(gif_path)
        # 60 frames per second
        imageio.mimwrite(gif_path, frames, duration=16.66)

    def report(
        self, reward: SupportsFloat, done: bool, model_losses: Dict[str, Any] = None
    ) -> None:
        """!
        Keep track of the last episodic rewards, episode length, and time elapse since last training iteration.
        @param reward: the current reward
        @param done: whether the episode ended
        @param model_losses: the current variational free energy, log-likelihood and KL-divergence
        """

        # Keep track of the memory usage of the program.
        info = self.process.memory_info()
        self.residential_mem.append(info.rss)

        # Keep track of time elapsed since last training iteration.
        now = time.time() * 1000
        if self.last_time == -1 and len(self.time_elapsed) != 0:
            # Ensure a smooth time curve upon reload.
            self.time_elapsed.append(self.time_elapsed[-1])
        elif self.last_time is not None:
            self.time_elapsed.append(now - self.last_time)
        self.last_time = now

        # Keep track of the current episodic reward.
        self.curr_ep_reward += float(reward)
        self.curr_ep_length += 1

        # Keep track of the model-based metrics.
        if model_losses is not None:
            self.vfe_losses.append(model_losses["vfe"].item())
            self.betas.append(model_losses["beta"])
            self.log_likelihoods.append(model_losses["log_likelihood"].item())
            self.kl_divergences.append(model_losses["kl_divergence"].item())
            if "efe_loss" in model_losses.keys():
                self.efe_losses.append(model_losses["efe_loss"].item())

        # Keep track of the current episodic reward and episode length.
        if done:
            self.episodic_rewards.append(self.curr_ep_reward)
            self.curr_ep_reward = 0
            self.episode_lengths.append(self.curr_ep_length)
            self.curr_ep_length = 0

    def log_performance_in_tensorboard(self) -> None:
        """!
        Log the agent performance in tensorboard, if the internal queue
        """

        # Log the mean time elapsed between two training iterations.
        if len(self.residential_mem) >= 2:
            self.log_mean_metric("mean_memory_gb", self.residential_mem, 1e9)

        # Log the mean time elapsed between two training iterations.
        if len(self.time_elapsed) >= 2:
            self.log_mean_metric("mean_time_elapsed_ms", self.time_elapsed)

        # Log the mean episodic reward.
        if len(self.episodic_rewards) >= 2:
            self.log_mean_metric("mean_episodic_reward", self.episodic_rewards)

        # Log the mean episode length.
        if len(self.episode_lengths) >= 2:
            self.log_mean_metric("mean_episode_length", self.episode_lengths)

        # Log the mean variational free energy, log-likelihood, and
        # KL-divergence.
        if len(self.vfe_losses) >= 2:
            self.log_mean_metric("variational_free_energy", self.vfe_losses)
            self.log_mean_metric("beta", self.betas)
            self.log_mean_metric("log_likelihood", self.log_likelihoods)
            self.log_mean_metric("kl_divergence", self.kl_divergences)

        # Log the expected free energy loss, if needed.
        if len(self.efe_losses) >= 2:
            self.log_mean_metric("efe_loss", self.efe_losses)

    def log_mean_metric(self, name: str, values: deque, scale: float = 1) -> None:
        """!
        Log the mean metric value in TensorBoard.
        :param name: the metric name
        :param values: the metric values
        :param scale: a coefficient by which all values must be divided
        """
        self.writer.add_scalar(name, np.mean(list(values)) / scale, self.current_step)

    @staticmethod
    def get_latest_checkpoint(regex: str = r"model_\d+.pt") -> Optional[str]:
        """!
        Get the latest checkpoint file matching the regex.
        @param regex: the regex checking whether a file name is a valid checkpoint file
        @return None if an error occurred, else the path to the latest checkpoint
        """

        # If the path is not a directory or does not exist, return without
        # trying to load the checkpoint.
        directory = os.environ["CHECKPOINT_DIRECTORY"]
        if not exists(directory) or not isdir(directory):
            logging.warning(f"Directory not found: {directory}")
            return None

        # If the directory does not contain any files,
        # return without trying to load the checkpoint.
        files = [
            file for file in os.listdir(directory) if isfile(join(directory, file))
        ]
        if len(files) == 0:
            logging.warning(f"No checkpoint found in directory: {directory}")
            return None

        # Retrieve the file whose name contain the largest number.
        # This number must be the time step at which the agent was saved.
        max_number = -math.inf
        file = None
        for current_file in files:

            # Get the number of training steps of the current checkpoint file.
            if len(re.findall(regex, current_file)) == 0:
                continue
            current_number = max(
                [int(number) for number in re.findall(r"\d+", current_file)]
            )

            # Track the checkpoint with the largest number of training steps.
            if current_number > max_number:
                max_number = current_number
                file = join(directory, current_file)

        return file

    @staticmethod
    def get_replay_buffer(
        capacity: int,
        batch_size: int,
        replay_type: ReplayType,
        omega: float,
        omega_is: float,
        n_steps: int,
        gamma: float = 1.0,
    ) -> ReplayBuffer:
        """!
        Retrieve the constructor of the replay buffer requested as parameters.
        @param capacity: the number of experiences the replay buffer can store
        @param batch_size: the size of the batches sampled from the replay buffer
        @param replay_type: the type of replay buffer
        @param omega: the prioritization exponent
        @param omega_is: the important sampling exponent
        @param n_steps: the number of steps for which rewards are accumulated in multistep Q-learning
        @param gamma: the discount factor
        @return the constructor of the replay buffer
        """
        m_args = {"n_steps": n_steps, "gamma": gamma}
        p_args = {"initial_priority": 1e9, "omega": omega, "omega_is": omega_is}
        args = m_args | p_args
        buffer = {
            ReplayType.DEFAULT: ReplayBuffer,
            ReplayType.PRIORITIZED: partial(ReplayBuffer, args=p_args),
            ReplayType.MULTISTEP: partial(ReplayBuffer, args=m_args),
            ReplayType.MULTISTEP_PRIORITIZED: partial(ReplayBuffer, args=args),
        }[replay_type]
        return buffer(capacity=capacity, batch_size=batch_size)
