import abc
import logging
import math
import os
import re
import time
from abc import ABC
from collections import deque
from os.path import exists, isdir, isfile, join

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import imageio
from PIL import Image

import benchmarks
from benchmarks.helpers.FileSystem import FileSystem


class AgentInterface(ABC):
    """
    The interface that all agents must implement.
    """

    def __init__(self, training=True):
        """
        Create an agent.
        :param training: True if the agent is being training, False otherwise
        """

        # Retrieve the device to use for training.
        self.device = benchmarks.device()

        # The number of episodes for which to track metrics.
        self.max_queue_len = 100

        # The number of training steps performed.
        self.current_step = 0

        # Create the queue containing the last episodic rewards.
        self.episodic_rewards = deque(maxlen=self.max_queue_len)
        self.current_episodic_reward = 0

        # Create the queue containing the time elapsed between training iterations.
        self.time_elapsed = deque(maxlen=self.max_queue_len)
        self.last_time = None

        # Create the queue containing the last episode lengths.
        self.episode_lengths = deque(maxlen=self.max_queue_len)
        self.current_episode_length = 0

        # Create the summary writer for monitoring with TensorBoard.
        self.writer = SummaryWriter(os.environ["TENSORBOARD_DIRECTORY"]) if training is True else None

    @abc.abstractmethod
    def step(self, obs):
        """
        Select the next action to perform in the environment.
        :param obs: the observation available to make the decision
        :return: the next action to perform
        """
        ...

    @abc.abstractmethod
    def train(self, env):
        """
        Train the agent in the gym environment passed as parameters
        :param env: the gym environment
        """
        ...

    @abc.abstractmethod
    def load(self, checkpoint_name=None):
        """
        Load an agent from the filesystem.
        :param checkpoint_name: the name of the checkpoint to load
        """
        ...

    @abc.abstractmethod
    def save(self, checkpoint_name):
        """
        Save the agent on the filesystem.
        :param checkpoint_name: the name of the checkpoint in which to save the agent
        """
        ...

    def demo(self, env, gif_name, max_frames=10000):
        """
        Train the agent in the gym environment passed as parameters
        :param env: the gym environment
        :param gif_name: the name of the GIF file in which to save the demo
        :param max_frames: the maximum number of frames to include in the GIF file
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
        imageio.mimwrite(gif_path, frames, fps=60)

    def report(self, reward, done):
        """
        Keep track of the last episodic rewards, episode length, and time elapse since last training iteration.
        :param reward: the current reward
        :param done: whether the episode ended
        """

        # Keep track of time elapsed since last training iteration.
        now = time.time() * 1000
        if self.last_time is not None:
            self.time_elapsed.append(now - self.last_time)
        self.last_time = now

        # Keep track of current episodic reward.
        self.current_episodic_reward += reward
        self.current_episode_length += 1

        # If the episode ended, keep track of the current episodic reward and episode length.
        if done:
            self.episodic_rewards.append(self.current_episodic_reward)
            self.current_episodic_reward = 0
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_length = 0

    def log_performance_in_tensorboard(self):
        """
        Log the agent performance in tensorboard, if the internal queue
        """

        # Log the mean time elapsed between two training iterations.
        if len(self.time_elapsed) >= 2:
            self.writer.add_scalar("mean_time_elapsed_ms", np.mean(list(self.time_elapsed)), self.current_step)

        # Log the mean episodic reward.
        if len(self.episodic_rewards) >= 2:
            self.writer.add_scalar("mean_episodic_reward", np.mean(list(self.episodic_rewards)), self.current_step)

        # Log the mean episode length.
        if len(self.episode_lengths) >= 2:
            self.writer.add_scalar("mean_episode_length", np.mean(list(self.episode_lengths)), self.current_step)

    @staticmethod
    def get_latest_checkpoint(regex=r"model_\d+.pt"):
        """
        Get the latest checkpoint file matching the regex.
        :param regex: the regex checking whether a file name is a valid checkpoint file
        :return: None if an error occurred, else the path to the latest checkpoint
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
