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
from typing import Dict, Any, Callable, SupportsFloat, Iterator, Optional, List, Union, Tuple

import psutil

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import imageio
from PIL import Image
from torch import optim
from gymnasium import Env

import relab
from relab.agents.memory.ReplayBuffer import ReplayBuffer
from relab.helpers.FileSystem import FileSystem
from relab.helpers.Typing import ActionType, Checkpoint, Parameter, Optimizer, ObservationType


class ReplayType(IntEnum):
    """!
    The type of replay buffer supported by the agents.
    """

    ## @var DEFAULT
    # Standard replay buffer with uniform sampling.
    DEFAULT = 0

    ## @var PRIORITIZED
    # Prioritized experience replay buffer that samples transitions based on their associated loss values.
    PRIORITIZED = 1

    ## @var MULTISTEP
    # Replay buffer that stores n-step transitions for multistep Q-learning.
    MULTISTEP = 2

    ## @var MULTISTEP_PRIORITIZED
    # Combination of prioritized experience replay and n-step transitions.
    MULTISTEP_PRIORITIZED = 3


class AgentInterface(ABC):
    """!
    @brief The interface that all agents must implement.
    """

    def __init__(self, training : bool = True) -> None:
        """!
        Create an agent.
        @param training: True if the agent is being training, False otherwise
        """

        ## @var training
        # Flag indicating whether the agent is in training mode.
        self.training = training

        ## @var device
        # The device (CPU/GPU) used for training computations.
        self.device = relab.device()

        ## @var max_queue_len
        # Maximum length for metric tracking queues (e.g., rewards, losses).
        self.max_queue_len = 100

        ## @var current_step
        # Counter tracking the number of training steps performed.
        self.current_step = 0

        ## @var vfe_losses
        # Queue containing the last variational free energy loss values.
        self.vfe_losses = deque(maxlen=self.max_queue_len)

        ## @var betas
        # Queue containing the last beta values for variational inference.
        self.betas = deque(maxlen=self.max_queue_len)

        ## @var log_likelihoods
        # Queue containing the last log-likelihood values.
        self.log_likelihoods = deque(maxlen=self.max_queue_len)

        ## @var kl_divergences
        # Queue containing the last KL-divergence values.
        self.kl_divergences = deque(maxlen=self.max_queue_len)

        ## @var process
        # Object representing the current process, used to track memory usage.
        self.process = psutil.Process()

        ## @var virtual_memory
        # Queue tracking virtual memory usage over time.
        self.virtual_memory = deque(maxlen=self.max_queue_len)

        ## @var residential_memory
        # Queue tracking residential memory usage over time.
        self.residential_memory = deque(maxlen=self.max_queue_len)

        ## @var episodic_rewards
        # Queue containing the last episodic reward values.
        self.episodic_rewards = deque(maxlen=self.max_queue_len)

        ## @var current_episodic_reward
        # Accumulator for the current episode's reward.
        self.current_episodic_reward = 0

        ## @var time_elapsed
        # Queue containing the time elapsed between consecutive training iterations.
        self.time_elapsed = deque(maxlen=self.max_queue_len)

        ## @var last_time
        # Timestamp of the last training iteration.
        self.last_time = None

        ## @var episode_lengths
        # Queue containing the lengths of recent episodes.
        self.episode_lengths = deque(maxlen=self.max_queue_len)

        ## @var current_episode_length
        # Counter for the current episode's length.
        self.current_episode_length = 0

        ## @var writer
        # TensorBoard summary writer for logging training metrics.
        self.writer = SummaryWriter(os.environ["TENSORBOARD_DIRECTORY"]) if training is True else None

    @abc.abstractmethod
    def step(self, obs : ObservationType) -> ActionType:
        """!
        Select the next action to perform in the environment.
        @param obs: the observation available to make the decision
        @return the next action to perform
        """
        ...

    @abc.abstractmethod
    def train(self, env : Env) -> None:
        """!
        Train the agent in the gym environment passed as parameters
        @param env: the gym environment
        """
        ...

    def load(
        self,
        checkpoint_name : Optional[str] = None,
        buffer_checkpoint_name : Optional[str] = None
    ) -> Tuple[str, Checkpoint]:
        """!
        Load an agent from the filesystem.
        @param checkpoint_name: the name of the agent checkpoint to load
        @param buffer_checkpoint_name: the name of the replay buffer checkpoint to load (None for default name)
        @return a tuple containing the checkpoint path and the checkpoint object
        """

        # Retrieve the full agent checkpoint path.
        if checkpoint_name is None:
            checkpoint_path = self.get_latest_checkpoint()
        else:
            checkpoint_path = join(os.environ["CHECKPOINT_DIRECTORY"], checkpoint_name)

        # Check if the checkpoint can be loaded.
        if checkpoint_path is None:
            logging.info("Could not load the agent from the file system.")
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), "checkpoint.pt")

        # Load the checkpoint from the file system.
        logging.info("Loading agent from the following file: " + checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Load the class attributes from the checkpoint.
        self.training = self.safe_load(checkpoint, "training")
        self.current_step = self.safe_load(checkpoint, "current_step")
        self.max_queue_len = self.safe_load(checkpoint, "max_queue_len")
        self.current_episodic_reward = self.safe_load(checkpoint, "current_episodic_reward")
        self.last_time = self.safe_load(checkpoint, "last_time")
        self.current_episode_length = self.safe_load(checkpoint, "current_episode_length")
        self.vfe_losses = self.safe_load(checkpoint, "vfe_losses")
        self.betas = self.safe_load(checkpoint, "betas")
        self.log_likelihoods = self.safe_load(checkpoint, "log_likelihoods")
        self.kl_divergences = self.safe_load(checkpoint, "kl_divergences")
        self.virtual_memory = self.safe_load(checkpoint, "virtual_memory")
        self.residential_memory = self.safe_load(checkpoint, "residential_memory")
        self.episodic_rewards = self.safe_load(checkpoint, "episodic_rewards")
        self.time_elapsed = self.safe_load(checkpoint, "time_elapsed")
        self.episode_lengths = self.safe_load(checkpoint, "episode_lengths")
        return checkpoint_path, checkpoint

    def as_dict(self):
        """"
        Convert the agent into a dictionary that can be saved on the filesystem.
        @return the dictionary
        """
        return {
            "training": self.training,
            "current_step": self.current_step,
            "max_queue_len": self.max_queue_len,
            "current_episodic_reward": self.current_episodic_reward,
            "last_time": self.last_time,
            "current_episode_length": self.current_episode_length,
            "vfe_losses": self.vfe_losses,
            "betas": self.betas,
            "log_likelihoods": self.log_likelihoods,
            "kl_divergences": self.kl_divergences,
            "virtual_memory": self.virtual_memory,
            "residential_memory": self.residential_memory,
            "episodic_rewards": self.episodic_rewards,
            "time_elapsed": self.time_elapsed,
            "episode_lengths": self.episode_lengths
        }

    @abc.abstractmethod
    def save(self, checkpoint_name : str, buffer_checkpoint_name : Optional[str] = None) -> None:
        """!
        Save the agent on the filesystem.
        @param checkpoint_name: the name of the checkpoint in which to save the agent
        @param buffer_checkpoint_name: the name of the checkpoint to save the replay buffer (None for default name)
        """
        ...

    def demo(self, env : Env, gif_name : str, max_frames : int = 10000) -> None:
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
            frames.append(Image.fromarray(env.render()))

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
        imageio.mimwrite(gif_path, frames, duration=16.66)  # 60 frames per second

    def report(self, reward : SupportsFloat, done : bool, model_losses : Dict[str, Any] = None) -> None:
        """!
        Keep track of the last episodic rewards, episode length, and time elapse since last training iteration.
        @param reward: the current reward
        @param done: whether the episode ended
        @param model_losses: the current variational free energy, log-likelihood and KL-divergence
        """

        # Keep track of the memory usage of the program.
        info = self.process.memory_info()
        self.virtual_memory.append(info.vms)
        self.residential_memory.append(info.rss)

        # Keep track of time elapsed since last training iteration.
        now = time.time() * 1000
        if self.last_time is not None:
            self.time_elapsed.append(now - self.last_time)
        self.last_time = now

        # Keep track of the current episodic reward.
        self.current_episodic_reward += float(reward)
        self.current_episode_length += 1

        # Keep track of the current variational free energy, log-likelihood and KL-divergence.
        if model_losses is not None:
            self.vfe_losses.append(model_losses["vfe"].item())
            self.betas.append(model_losses["beta"])
            self.log_likelihoods.append(model_losses["log_likelihood"].item())
            self.kl_divergences.append(model_losses["kl_divergence"].item())

        # If the episode ended, keep track of the current episodic reward and episode length.
        if done:
            self.episodic_rewards.append(self.current_episodic_reward)
            self.current_episodic_reward = 0
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_length = 0

    def log_performance_in_tensorboard(self) -> None:
        """!
        Log the agent performance in tensorboard, if the internal queue
        """

        # Log the mean time elapsed between two training iterations.
        if len(self.virtual_memory) >= 2:
            self.writer.add_scalar("mean_virtual_memory_gb", np.mean(list(self.virtual_memory)) / 1e9, self.current_step)
            self.writer.add_scalar("mean_residential_memory_gb", np.mean(list(self.residential_memory)) / 1e9, self.current_step)

        # Log the mean time elapsed between two training iterations.
        if len(self.time_elapsed) >= 2:
            self.writer.add_scalar("mean_time_elapsed_ms", np.mean(list(self.time_elapsed)), self.current_step)

        # Log the mean episodic reward.
        if len(self.episodic_rewards) >= 2:
            self.writer.add_scalar("mean_episodic_reward", np.mean(list(self.episodic_rewards)), self.current_step)

        # Log the mean episode length.
        if len(self.episode_lengths) >= 2:
            self.writer.add_scalar("mean_episode_length", np.mean(list(self.episode_lengths)), self.current_step)

        # Log the mean variational free energy, log-likelihood, and KL-divergence.
        if len(self.vfe_losses) >= 2:
            self.writer.add_scalar("variational_free_energy", np.mean(list(self.vfe_losses)), self.current_step)
            self.writer.add_scalar("beta", np.mean(list(self.betas)), self.current_step)
            self.writer.add_scalar("log_likelihood", np.mean(list(self.log_likelihoods)), self.current_step)
            self.writer.add_scalar("kl_divergence", np.mean(list(self.kl_divergences)), self.current_step)

    @staticmethod
    def get_latest_checkpoint(regex : str = r"model_\d+.pt") -> Optional[str]:
        """!
        Get the latest checkpoint file matching the regex.
        @param regex: the regex checking whether a file name is a valid checkpoint file
        @return None if an error occurred, else the path to the latest checkpoint
        """
        
        # If the path is not a directory or does not exist, return without trying to load the checkpoint.
        directory = os.environ["CHECKPOINT_DIRECTORY"]
        if not exists(directory) or not isdir(directory):
            logging.warning("The following directory was not found: " + directory)
            return None

        # If the directory does not contain any files, return without trying to load the checkpoint.
        files = [file for file in os.listdir(directory) if isfile(join(directory, file))]
        if len(files) == 0:
            logging.warning("No checkpoint found in directory: " + directory)
            return None

        # Retrieve the file whose name contain the largest number.
        # This number is assumed to be the time step at which the agent was saved.
        max_number = - math.inf
        file = None
        for current_file in files:
            
            # Retrieve the number of training steps of the current checkpoint file.
            if len(re.findall(regex, current_file)) == 0:
                continue
            current_number = max([int(number) for number in re.findall(r"\d+", current_file)])

            # Remember the checkpoint file with the highest number of training steps.
            if current_number > max_number:
                max_number = current_number
                file = join(directory, current_file)

        return file

    @staticmethod
    def get_replay_buffer(replay_type : ReplayType, omega : float, omega_is : float, n_steps : int, gamma : float = 1.0) -> Callable:
        """!
        Retrieve the constructor of the replay buffer requested as parameters.
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
        return {
            ReplayType.DEFAULT: ReplayBuffer,
            ReplayType.PRIORITIZED: partial(ReplayBuffer, args=p_args),
            ReplayType.MULTISTEP: partial(ReplayBuffer, args=m_args),
            ReplayType.MULTISTEP_PRIORITIZED: partial(ReplayBuffer, args=args),
        }[replay_type]

    @staticmethod
    def safe_load(checkpoint : Checkpoint, key : str) -> Any:
        """!
        Load the value corresponding to the key in the checkpoint.
        @param checkpoint: the checkpoint
        @param key: the key
        @return the value, or None if the key is not in the checkpoint
        """
        if key not in checkpoint.keys():
            return None
        return checkpoint[key]

    @staticmethod
    def safe_load_optimizer(
        checkpoint : Checkpoint,
        params : Union[Iterator[Parameter], List[Parameter]],
        learning_rate : float,
        adam_eps : float
    ) -> Optimizer:
        """!
        Load the Adam optimizer from the checkpoint safely.
        @param checkpoint: the checkpoint
        @param params: the parameters that the Adam optimizer must optimize
        @param learning_rate: the learning rate
        @param adam_eps: the epsilon parameter of the Adam optimizer
        @return the loaded Adam optimizer
        """
        optimizer = optim.Adam(params, lr=learning_rate, eps=adam_eps)
        if "optimizer" not in checkpoint.keys():
            logging.info("Optimizer internal states could not be loaded from the checkpoint: key 'optimizer' not found.")
            return optimizer
        optimizer.load_state_dict(checkpoint["optimizer"])
        return optimizer
