import logging
from datetime import datetime
from os.path import join
from typing import Tuple

import numpy as np
import relab
import torch
from gymnasium import Env

from relab.agents.AgentInterface import AgentInterface
from relab.helpers.FileSystem import FileSystem
from relab.helpers.Serialization import safe_load
from relab.helpers.Typing import ActionType, Checkpoint, ObservationType


class Random(AgentInterface):
    """!
    @brief Implements an agent taking random actions.
    """

    def __init__(self, n_actions: int = 18, training: bool = True) -> None:
        """!
        Create an agent taking random actions.
        @param n_actions: the number of actions available to the agent
        @param training: True if the agent is being training, False otherwise
        """

        # Call the parent constructor.
        super().__init__(training=training)

        # @var n_actions
        # Number of possible actions available to the agent.
        self.n_actions = n_actions

    def step(self, obs: ObservationType) -> ActionType:
        """!
        Select the next action to perform in the environment.
        @param obs: the observation available to make the decision
        @return the next action to perform
        """
        return np.random.choice(self.n_actions)

    def train(self, env: Env) -> None:
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

    def load(
        self, checkpoint_name: str = "", buffer_checkpoint_name: str = ""
    ) -> Tuple[str, Checkpoint]:
        """!
        Load an agent from the filesystem.
        @param checkpoint_name: the name of the agent checkpoint to load
        @param buffer_checkpoint_name: the name of the replay buffer checkpoint to load (None for default name)
        @return a tuple containing the checkpoint path and the checkpoint object
        """
        # @cond IGNORED_BY_DOXYGEN
        try:
            # Call the parent load function.
            checkpoint_path, checkpoint = super().load(
                checkpoint_name, buffer_checkpoint_name
            )

            # Load the class attributes from the checkpoint.
            self.n_actions = safe_load(checkpoint, "n_actions")
            return checkpoint_path, checkpoint

        # Catch the exception raise if the checkpoint could not be located.
        except FileNotFoundError:
            return "", None
        # @endcond

    def as_dict(self):
        """!
        Convert the agent into a dictionary that can be saved on the filesystem.
        @return the dictionary
        """
        return {"n_actions": self.n_actions} | super().as_dict()

    def save(self, checkpoint_name: str, buffer_checkpoint_name: str = "") -> None:
        """!
        Save the agent on the filesystem.
        @param checkpoint_name: the name of the checkpoint in which to save the agent
        @param buffer_checkpoint_name: the name of the checkpoint to save the replay buffer (None for default name)
        """
        # @cond IGNORED_BY_DOXYGEN
        # Create the checkpoint directory and file, if they do not exist.
        checkpoint_path = join(self.checkpoint_dir, checkpoint_name)
        FileSystem.create_directory_and_file(checkpoint_path)

        # Save the model.
        logging.info(f"Saving agent to the following file: {checkpoint_path}")
        torch.save(self.as_dict(), checkpoint_path)
        # @endcond
