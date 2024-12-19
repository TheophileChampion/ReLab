import logging
import os
from os.path import join

import torch

from benchmarks.agents.AgentInterface import ReplayType
from benchmarks.agents.Random import Random

from benchmarks.helpers.FileSystem import FileSystem


class VAE(Random):
    """
    Implement an agent taking random actions, and learning a world model using the Variational Auto-Encoder from:
    Kingma Diederi, and Welling Max. Auto-Encoding Variational Bayes.
    International Conference on Learning Representations, 2014.
    """

    def __init__(
        self, learning_starts=200000, n_actions=18, training=True,
        replay_type=ReplayType.DEFAULT, buffer_size=1000000, batch_size=32, n_steps=1, omega=1.0, omega_is=1.0
    ):
        """
        Create a Variational Auto-Encoder agent taking random actions.
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
        super().__init__(
            learning_starts=learning_starts, n_actions=n_actions, training=training,  replay_type=replay_type,
            buffer_size=buffer_size, batch_size=batch_size, n_steps=n_steps, omega=omega, omega_is=omega_is,
        )

    def learn(self):
        """
        TODO
        Perform one step of gradient descent on the world model.
        """
        ...

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
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

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
