import logging
import os
from datetime import datetime
from os.path import join

import torch

import relab
from relab.agents.AgentInterface import AgentInterface, ReplayType
import numpy as np

from relab.helpers.FileSystem import FileSystem


class Random(AgentInterface):
    """!
    Implement an agent taking random actions.
    """

    def __init__(self, n_actions=18):
        """!
        Create an agent taking random actions.
        @param n_actions: the number of actions available to the agent
        """

        # Call the parent constructor.
        super().__init__(training=training)

        ## @var n_actions
        # Number of possible actions available to the agent.
        self.n_actions = n_actions

    def step(self, obs):
        """!
        Select the next action to perform in the environment.
        @param obs: the observation available to make the decision
        @return the next action to perform
        """
        return np.random.choice(self.n_actions)

    def train(self, env):
        """!
        Train the agent in the gym environment passed as parameters
        @param env: the gym environment
        """
        # @cond IGNORED_BY_DOXYGEN

        # Retrieve the initial observation from the environment.
        obs, _ = env.reset()

        # Train the agent.
        config = relab.config()
        logging.info(f"Start the training at {datetime.now()}")
        while self.current_step < config["max_n_steps"]:

            # Select an action.
            action = self.step(obs.to(self.device))

            # Execute the action in the environment.
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

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
        # @endcond

    def load(self, checkpoint_name=None, buffer_checkpoint_name=None):
        """!
        Load an agent from the filesystem.
        @param checkpoint_name: the name of the agent checkpoint to load
        @param buffer_checkpoint_name: the name of the replay buffer checkpoint to load (None for default name)
        """
        # @cond IGNORED_BY_DOXYGEN

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
        self.n_actions = self.safe_load(checkpoint, "n_actions")
        self.current_step = self.safe_load(checkpoint, "current_step")
        # @endcond

    def save(self, checkpoint_name, buffer_checkpoint_name=None):
        """!
        Save the agent on the filesystem.
        @param checkpoint_name: the name of the checkpoint in which to save the agent
        @param buffer_checkpoint_name: the name of the checkpoint to save the replay buffer (None for default name)
        """
        # @cond IGNORED_BY_DOXYGEN

        # Create the checkpoint directory and file, if they do not exist.
        checkpoint_path = join(os.environ["CHECKPOINT_DIRECTORY"], checkpoint_name)
        FileSystem.create_directory_and_file(checkpoint_path)

        # Save the model.
        logging.info("Saving agent to the following file: " + checkpoint_path)
        torch.save({
            "n_actions": self.n_actions,
            "current_step": self.current_step,
        }, checkpoint_path)
        # @endcond
