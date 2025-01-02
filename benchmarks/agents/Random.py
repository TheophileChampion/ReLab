import logging
import os
from datetime import datetime
from os.path import join

import torch

import benchmarks
from benchmarks.agents.AgentInterface import AgentInterface, ReplayType
from benchmarks.agents.memory.ReplayBuffer import Experience
import numpy as np

from benchmarks.helpers.FileSystem import FileSystem


class Random(AgentInterface):
    """
    Implement an agent taking random actions.
    """

    def __init__(
        self, learning_starts=200000, n_actions=18, training=False,
        replay_type=ReplayType.DEFAULT, buffer_size=1000000, batch_size=32, n_steps=1, omega=1.0, omega_is=1.0
    ):
        """
        Create an agent taking random actions.
        :param learning_starts: the step at which learning starts
        :param n_actions: the number of actions available to the agent
        :param training: True if the agent is being trained, False otherwise
        :param replay_type: the type of replay buffer
        :param buffer_size: the size of the replay buffer
        :param batch_size: the size of the batches sampled from the replay buffer
        :param n_steps: the number of steps for which rewards are accumulated in multistep Q-learning
        :param omega: the prioritization exponent
        :param omega_is: the important sampling exponent
        """

        # Call the parent constructor.
        super().__init__(training=training)

        # Store the agent's parameters.
        self.learning_starts = learning_starts
        self.n_actions = n_actions
        self.training = training
        self.replay_type = replay_type
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.omega = omega
        self.omega_is = omega_is

        # Create the replay buffer.
        replay_buffer = self.get_replay_buffer(self.replay_type, self.omega, self.omega_is, self.n_steps)
        self.buffer = replay_buffer(capacity=self.buffer_size, batch_size=self.batch_size) if self.training else None

    def step(self, obs):
        """
        Select the next action to perform in the environment.
        :param obs: the observation available to make the decision
        :return: the next action to perform
        """
        return np.random.choice(self.n_actions)

    def train(self, env):
        """
        Train the agent in the gym environment passed as parameters
        :param env: the gym environment
        """

        # Retrieve the initial observation from the environment.
        obs, _ = env.reset()

        # Train the agent.
        config = benchmarks.config()
        logging.info(f"Start the training at {datetime.now()}")
        while self.current_step < config["max_n_steps"]:

            # Select an action.
            action = self.step(obs.to(self.device))

            # Execute the action in the environment.
            old_obs = obs
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Add the experience to the replay buffer.
            if self.training is True:
                self.buffer.append(Experience(old_obs, action, reward, done, obs))

            # Save the agent (if needed).
            if self.current_step % config["checkpoint_frequency"] == 0:
                self.save(f"model_{self.current_step}.pt")

            # Log the mean episodic reward in tensorboard (if needed).
            self.report(reward, done)
            if self.current_step % config["tensorboard_log_interval"] == 0:
                self.log_performance_in_tensorboard()

            # Reset the environment when a trial ends.
            if done:
                obs, _ = env.reset()

            # Increase the number of training steps done.
            self.current_step += 1

        # Save the final version of the model.
        self.save(f"model_{config['max_n_steps']}.pt")

        # Close the environment.
        env.close()

    def load(self, checkpoint_name=None):
        """
        Load an agent from the filesystem.
        :param checkpoint_name: the name of the checkpoint to load
        """

        # Retrieve the full checkpoint path.
        if checkpoint_name is None:
            checkpoint_path = self.get_latest_checkpoint()
        else:
            checkpoint_path = join(os.environ["CHECKPOINT_DIRECTORY"], checkpoint_name)

        # Check if the checkpoint can be loaded.
        if checkpoint_path is None:
            logging.info("Could not load the agent from the file system.")
            return
        
        # Load the checkpoint from the file system.
        logging.info("Loading agent from the following file: " + checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Update the agent's parameters using the checkpoint.
        self.buffer_size = self.safe_load(checkpoint, "buffer_size")
        self.batch_size = self.safe_load(checkpoint, "batch_size")
        self.learning_starts = self.safe_load(checkpoint, "learning_starts")
        self.n_actions = self.safe_load(checkpoint, "n_actions")
        self.n_steps = self.safe_load(checkpoint, "n_steps")
        self.omega = self.safe_load(checkpoint, "omega")
        self.omega_is = self.safe_load(checkpoint, "omega_is")
        self.replay_type = self.safe_load(checkpoint, "replay_type")
        self.training = self.safe_load(checkpoint, "training")
        self.current_step = self.safe_load(checkpoint, "current_step")

        # Update the replay buffer.
        replay_buffer = self.get_replay_buffer(self.replay_type, self.omega, self.omega_is, self.n_steps)
        self.buffer = replay_buffer(capacity=self.buffer_size, batch_size=self.batch_size) if self.training else None

    def save(self, checkpoint_name):
        """
        Save the agent on the filesystem.
        :param checkpoint_name: the name of the checkpoint in which to save the agent
        """
        
        # Create the checkpoint directory and file, if they do not exist.
        checkpoint_path = join(os.environ["CHECKPOINT_DIRECTORY"], checkpoint_name)
        FileSystem.create_directory_and_file(checkpoint_path)

        # Save the model.
        logging.info("Saving agent to the following file: " + checkpoint_path)
        torch.save({
            "buffer_size": self.buffer_size,
            "batch_size": self.batch_size,
            "learning_starts": self.learning_starts,
            "n_actions": self.n_actions,
            "n_steps": self.n_steps,
            "omega": self.omega,
            "omega_is": self.omega_is,
            "replay_type": self.replay_type,
            "training": self.training,
            "current_step": self.current_step,
        }, checkpoint_path)
