import abc
import logging
import os
import re
from datetime import datetime
from enum import IntEnum
from functools import partial
from os.path import join
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import relab
import torch
from gymnasium import Env
from matplotlib.figure import Figure
from relab.agents.AgentInterface import AgentInterface, ReplayType
from relab.agents.networks.DecoderNetwork import ContinuousDecoderNetwork
from relab.agents.networks.EncoderNetwork import ContinuousEncoderNetwork
from relab.agents.networks.TransitionNetwork import ContinuousTransitionNetwork
from relab.agents.schedule.PiecewiseLinearSchedule import PiecewiseLinearSchedule
from relab.cpp.agents.memory import Experience
from relab.helpers.MatPlotLib import MatPlotLib
from relab.helpers.Typing import (
    ActionType,
    AttributeNames,
    Checkpoint,
    Config,
    ObservationType,
)
from relab.helpers.VariationalInference import (
    bernoulli_log_likelihood,
    gaussian_log_likelihood,
)
from torch import Tensor, nn


class LikelihoodType(IntEnum):
    """!
    The type of likelihood supported by the model-based agents.
    """

    # @var GAUSSIAN
    # Gaussian likelihood for observations regarded as continuous real values.
    GAUSSIAN = 0

    # @var BERNOULLI
    # Bernoulli likelihood for binary or normalized observations.
    BERNOULLI = 1


class VariationalModel(AgentInterface):
    """!
    @brief The interface that all agents behaving randomly to learn a world models must implement.
    """

    def __init__(
        self,
        learning_starts: int = 200000,
        n_actions: int = 18,
        training: bool = False,
        likelihood_type: LikelihoodType = LikelihoodType.BERNOULLI,
        replay_type: ReplayType = ReplayType.DEFAULT,
        buffer_size: int = 1000000,
        batch_size: int = 32,
        n_steps: int = 1,
        omega: float = 1.0,
        omega_is: float = 1.0,
        n_continuous_vars: int = 10,
        learning_rate: float = 0.00001,
        adam_eps: float = 1.5e-4,
        beta_schedule: Any = None,
    ) -> None:
        """!
        Create an agent taking random actions.
        @param learning_starts: the step at which learning starts
        @param n_actions: the number of actions available to the agent
        @param training: True if the agent is being trained, False otherwise
        @param likelihood_type: the type of likelihood used by the world model
        @param replay_type: the type of replay buffer
        @param buffer_size: the size of the replay buffer
        @param batch_size: the size of the batches sampled from the replay buffer
        @param n_steps: the number of steps for which rewards are accumulated in multistep Q-learning
        @param omega: the prioritization exponent
        @param omega_is: the important sampling exponent
        @param n_continuous_vars: the number of continuous latent variables
        @param learning_rate: the learning rate
        @param adam_eps: the epsilon parameter of the Adam optimizer
        @param beta_schedule: the piecewise linear schedule of the KL-divergence weight of beta-VAE
        """

        # Call the parent constructor.
        get_buffer = partial(
            self.get_replay_buffer,
            buffer_size,
            batch_size,
            replay_type,
            omega,
            omega_is,
            n_steps,
        )
        super().__init__(get_buffer=get_buffer, n_actions=n_actions, training=training)

        # @var likelihood_type
        # The type of likelihood function used in the world model (Gaussian or
        # Bernoulli).
        self.likelihood_type = likelihood_type

        # @var learning_starts
        # The step count at which learning begins.
        self.learning_starts = learning_starts

        # @var replay_type
        # Type of experience replay buffer being used.
        self.replay_type = replay_type

        # @var buffer_size
        # Maximum number of transitions stored in the replay buffer.
        self.buffer_size = buffer_size

        # @var batch_size
        # Number of transitions sampled per learning update.
        self.batch_size = batch_size

        # @var n_continuous_vars
        # Number of continuous variables in the latent space.
        self.n_cont_vars = n_continuous_vars

        # @var n_steps
        # Number of steps for multi-step learning.
        self.n_steps = n_steps

        # @var omega
        # Exponent for prioritization in the replay buffer.
        self.omega = omega

        # @var omega_is
        # Exponent for importance sampling correction.
        self.omega_is = omega_is

        # @var learning_rate
        # Learning rate for the optimizer.
        self.learning_rate = learning_rate

        # @var adam_eps
        # Epsilon parameter for the Adam optimizer.
        self.adam_eps = adam_eps

        # @var beta_schedule
        # Schedule for the KL-divergence weight in beta-VAE.
        self.beta_schedule = [(0, 1.0)] if beta_schedule is None else beta_schedule

        # @var beta
        # Scheduler for the KL-divergence weight in beta-VAE.
        self.beta = PiecewiseLinearSchedule(self.beta_schedule)

        # @var likelihood_loss
        # Function computing the reconstruction loss.
        self.likelihood_loss = self.get_likelihood_loss(self.likelihood_type)

    @abc.abstractmethod
    def learn(self) -> Optional[Dict[str, Any]]:
        """!
        Perform one step of gradient descent on the world model.
        @return the loss of the sampled batch, None if no loss should be logged in Tensorboard
        """
        ...

    @abc.abstractmethod
    def draw_reconstructed_images(
        self, env: Env, model_index: int, grid_size: Tuple[int, int]
    ) -> Figure:
        """!
        Draw the ground truth and reconstructed images.
        @param env: the gym environment
        @param model_index: the index of the model for which images are generated
        @param grid_size: the size of the image grid to generate
        @return the figure containing the images
        """
        ...

    @staticmethod
    def get_likelihood_loss(likelihood_type: LikelihoodType) -> Callable:
        """!
        Retrieve the likelihood loss requested as parameters.
        @param likelihood_type: the type of likelihood used by the world model
        @return the loss
        """
        # @cond IGNORED_BY_DOXYGEN
        return {
            LikelihoodType.GAUSSIAN: gaussian_log_likelihood,
            LikelihoodType.BERNOULLI: bernoulli_log_likelihood,
        }[likelihood_type]
        # @endcond

    def get_encoder_network(self) -> nn.Module:
        """!
        Retrieve the encoder network.
        @return the encoder network
        """
        # @cond IGNORED_BY_DOXYGEN
        encoder = ContinuousEncoderNetwork(n_continuous_vars=self.n_cont_vars)
        encoder.train(self.training)
        encoder.to(self.device)
        return encoder
        # @endcond

    def get_decoder_network(self) -> nn.Module:
        """!
        Retrieve the decoder network.
        @return the decoder network
        """
        # @cond IGNORED_BY_DOXYGEN
        decoder = ContinuousDecoderNetwork(n_continuous_vars=self.n_cont_vars)
        decoder.train(self.training)
        decoder.to(self.device)
        return decoder
        # @endcond

    def get_transition_network(self) -> nn.Module:
        """!
        Retrieve the transition network.
        @return the transition network
        """
        # @cond IGNORED_BY_DOXYGEN
        transition = ContinuousTransitionNetwork(
            n_actions=self.n_actions, n_continuous_vars=self.n_cont_vars
        )
        transition.train(self.training)
        transition.to(self.device)
        return transition
        # @endcond

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
            old_obs = obs
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Add the experience to the replay buffer.
            if self.training is True:
                self.buffer.append(Experience(old_obs, action, reward, done, obs))

            # Perform one iteration of training (if needed).
            losses = None
            if self.training is True and self.current_step >= self.learning_starts:
                losses = self.learn()

            # Save the agent (if needed).
            if self.current_step % config["checkpoint_frequency"] == 0:
                self.save(f"model_{self.current_step}.pt")

            # Log the mean episodic reward in tensorboard (if needed).
            self.report(reward, done, model_losses=losses)
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

    def demo(self, env: Env, gif_name: str, max_frames: int = 10000) -> None:
        """!
        Demonstrate the agent policy in the gym environment passed as parameters
        @param env: the gym environment
        @param gif_name: the name of the GIF file in which to save the demo
        @param max_frames: the maximum number of frames to include in the GIF file
        """

        # Create the GIF file containing a demonstration of the agent's policy.
        super().demo(env, gif_name, max_frames)

        # Create a graph containing images generated by the world model.
        model_index = int(re.findall(r"\d+", gif_name)[0])
        fig = self.draw_reconstructed_images(env, model_index, grid_size=(6, 6))

        # Save the figure containing the ground truth and reconstructed images.
        file_name = gif_name.replace(".gif", "") + "_reconstructed_images.pdf"
        fig.savefig(join(os.environ["DEMO_DIRECTORY"], file_name))
        MatPlotLib.close()

    def reconstructed_image_from(self, decoder_output: Tensor) -> Tensor:
        """!
        Compute the reconstructed image from the decoder output.
        @param decoder_output: the tensor predicted by the decoder
        @return the reconstructed image
        """
        function = {
            LikelihoodType.GAUSSIAN: partial(torch.clamp, min=0, max=1),
            LikelihoodType.BERNOULLI: torch.sigmoid,
        }[self.likelihood_type]
        return function(decoder_output)

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

        # Call the parent load function.
        attr_names = [] if attr_names is None else list(attr_names)
        attr_names += list(VariationalModel.as_dict(self).keys())
        checkpoint = super().load(checkpoint_name, buffer_checkpoint_name, attr_names)

        # Ensure the beta parameter follows the loaded beta schedule.
        self.beta = PiecewiseLinearSchedule(self.beta_schedule)
        return checkpoint

    def as_dict(self) -> Config:
        """!
        Convert the agent into a dictionary that can be saved on the filesystem.
        @return the dictionary
        """
        return {
            "buffer_size": self.buffer_size,
            "batch_size": self.batch_size,
            "learning_starts": self.learning_starts,
            "n_actions": self.n_actions,
            "n_steps": self.n_steps,
            "omega": self.omega,
            "omega_is": self.omega_is,
            "replay_type": self.replay_type,
            "learning_rate": self.learning_rate,
            "adam_eps": self.adam_eps,
            "likelihood_type": self.likelihood_type,
            "beta_schedule": self.beta_schedule,
            "n_cont_vars": self.n_cont_vars,
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
        super().save(
            checkpoint_name,
            buffer_checkpoint_name,
            agent_conf | VariationalModel.as_dict(self),
        )
