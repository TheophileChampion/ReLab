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
from relab.agents.networks.DecoderNetworks import (
    ContinuousDecoderNetwork,
    DiscreteDecoderNetwork,
    MixedDecoderNetwork,
)
from relab.agents.networks.EncoderNetworks import (
    ContinuousEncoderNetwork,
    DiscreteEncoderNetwork,
    MixedEncoderNetwork,
)
from relab.agents.networks.TransitionNetworks import (
    ContinuousTransitionNetwork,
    DiscreteTransitionNetwork,
    MixedTransitionNetwork,
)
from relab.agents.schedule.ExponentialSchedule import ExponentialSchedule
from relab.agents.schedule.PiecewiseLinearSchedule import PiecewiseLinearSchedule
from relab.cpp.agents.memory import Experience
from relab.helpers.MatPlotLib import MatPlotLib
from relab.helpers.Serialization import safe_load
from relab.helpers.Typing import ActionType, Checkpoint, ObservationType
from relab.helpers.VariationalInference import (
    bernoulli_log_likelihood,
    continuous_reparameterization,
    discrete_reparameterization,
    gaussian_log_likelihood,
    mixed_reparameterization,
)
from torch import Tensor, nn


class LatentSpaceType(IntEnum):
    """!
    The type of latent spaces supported by the model-based agents.
    """

    # @var CONTINUOUS
    # Latent space with continuous variables following a Gaussian distribution.
    CONTINUOUS = 0

    # @var DISCRETE
    # Latent space with categorical variables using Gumbel-Softmax relaxation.
    DISCRETE = 1

    # @var MIXED
    # Hybrid latent space combining both continuous and discrete variables.
    MIXED = 2


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
        latent_space_type: LatentSpaceType = LatentSpaceType.CONTINUOUS,
        replay_type: ReplayType = ReplayType.DEFAULT,
        buffer_size: int = 1000000,
        batch_size: int = 32,
        n_steps: int = 1,
        omega: float = 1.0,
        omega_is: float = 1.0,
        n_continuous_vars: int = 10,
        n_discrete_vars: int = 20,
        n_discrete_vals: int = 10,
        learning_rate: float = 0.00001,
        adam_eps: float = 1.5e-4,
        beta_schedule: Any = None,
        tau_schedule: Any = None,
    ) -> None:
        """!
        Create an agent taking random actions.
        @param learning_starts: the step at which learning starts
        @param n_actions: the number of actions available to the agent
        @param training: True if the agent is being trained, False otherwise
        @param likelihood_type: the type of likelihood used by the world model
        @param latent_space_type: the type of latent space used by the world model
        @param replay_type: the type of replay buffer
        @param buffer_size: the size of the replay buffer
        @param batch_size: the size of the batches sampled from the replay buffer
        @param n_steps: the number of steps for which rewards are accumulated in multistep Q-learning
        @param omega: the prioritization exponent
        @param omega_is: the important sampling exponent
        @param n_continuous_vars: the number of continuous latent variables
        @param n_discrete_vars: the number of discrete latent variables
        @param n_discrete_vals: the number of values taken by all the discrete latent variables,
            or a list describing the number of values taken by each discrete latent variable
        @param learning_rate: the learning rate
        @param adam_eps: the epsilon parameter of the Adam optimizer
        @param beta_schedule: the piecewise linear schedule of the KL-divergence weight of beta-VAE
        @param tau_schedule: the exponential schedule of the temperature of the Gumbel-softmax
        """

        # Call the parent constructor.
        super().__init__(training=training)

        # @var likelihood_type
        # The type of likelihood function used in the world model (Gaussian or
        # Bernoulli).
        self.likelihood_type = likelihood_type

        # @var latent_space_type
        # The type of latent space used (continuous, discrete, or mixed).
        self.latent_type = latent_space_type

        # @var learning_starts
        # The step count at which learning begins.
        self.learning_starts = learning_starts

        # @var n_actions
        # Number of possible actions in the environment.
        self.n_actions = n_actions

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

        # @var n_discrete_vars
        # Number of discrete variables in the latent space.
        self.n_discrete_vars = n_discrete_vars

        # @var n_discrete_vals
        # Number of possible values for each discrete variable.
        self.n_discrete_vals = n_discrete_vals

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

        # @var tau_schedule
        # Schedule for the Gumbel-Softmax temperature.
        self.tau_schedule = (0.5, -3e-5) if tau_schedule is None else tau_schedule

        # @var tau
        # Scheduler for the Gumbel-Softmax temperature.
        self.tau = ExponentialSchedule(self.tau_schedule)

        # @var model_loss
        # Function computing the world model's loss.
        self.model_loss = self.get_model_loss(self.latent_type)

        # @var likelihood_loss
        # Function computing the reconstruction loss.
        self.likelihood_loss = self.get_likelihood_loss(self.likelihood_type)

        # @var reparameterize
        # Function for sampling latent states from the latent distribution.
        self.reparam = self.get_reparameterization(self.latent_type)

        # @var buffer
        # Experience replay buffer for storing transitions.
        replay_buffer = self.get_replay_buffer(
            self.replay_type, self.omega, self.omega_is, self.n_steps
        )
        self.buffer = (
            replay_buffer(self.buffer_size, self.batch_size) if self.training else None
        )

    @abc.abstractmethod
    def learn(self) -> Optional[Dict[str, Any]]:
        """!
        Perform one step of gradient descent on the world model.
        @return the loss of the sampled batch, None if no loss should be logged in Tensorboard
        """
        ...

    @abc.abstractmethod
    def continuous_vfe(
        self, obs: Tensor, actions: Tensor, next_obs: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """!
        Compute the variational free energy for a continuous latent space.
        @param obs: the observations at time t
        @param actions: the actions at time t
        @param next_obs: the observations at time t + 1
        @return a tuple containing the variational free energy, log-likelihood and KL-divergence
        """
        ...

    @abc.abstractmethod
    def discrete_vfe(
        self, obs: Tensor, actions: Tensor, next_obs: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """!
        Compute the variational free energy for a discrete latent space.
        @param obs: the observations at time t
        @param actions: the actions at time t
        @param next_obs: the observations at time t + 1
        @return a tuple containing the variational free energy, log-likelihood and KL-divergence
        """
        ...

    @abc.abstractmethod
    def mixed_vfe(
        self, obs: Tensor, actions: Tensor, next_obs: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """!
        Compute the variational free energy for a mixed latent space.
        @param obs: the observations at time t
        @param actions: the actions at time t
        @param next_obs: the observations at time t + 1
        @return a tuple containing the variational free energy, log-likelihood and KL-divergence
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

    def get_model_loss(self, latent_space_type: LatentSpaceType) -> Callable:
        """!
        Retrieve the model loss requested as parameters.
        @param latent_space_type: the type of latent spaces used by the world model
        @return the loss
        """
        # @cond IGNORED_BY_DOXYGEN
        return {
            LatentSpaceType.CONTINUOUS: self.continuous_vfe,
            LatentSpaceType.DISCRETE: self.discrete_vfe,
            LatentSpaceType.MIXED: self.mixed_vfe,
        }[latent_space_type]
        # @endcond

    @staticmethod
    def get_reparameterization(latent_space_type: LatentSpaceType) -> Callable:
        """!
        Retrieve the reparameterization function requested as parameters.
        @param latent_space_type: the type of latent spaces used by the world model
        @return the reparameterization function
        """
        # @cond IGNORED_BY_DOXYGEN
        return {
            LatentSpaceType.CONTINUOUS: continuous_reparameterization,
            LatentSpaceType.DISCRETE: discrete_reparameterization,
            LatentSpaceType.MIXED: mixed_reparameterization,
        }[latent_space_type]
        # @endcond

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

    def get_encoder_network(self, latent_space_type: LatentSpaceType) -> nn.Module:
        """!
        Retrieve the encoder network requested as parameters.
        @param latent_space_type: the type of latent spaces to use for the encoder network
        @return the encoder network
        """
        # @cond IGNORED_BY_DOXYGEN
        encoder = {
            LatentSpaceType.CONTINUOUS: partial(
                ContinuousEncoderNetwork, n_continuous_vars=self.n_cont_vars
            ),
            LatentSpaceType.DISCRETE: partial(
                DiscreteEncoderNetwork,
                n_discrete_vars=self.n_discrete_vars,
                n_discrete_vals=self.n_discrete_vals,
            ),
            LatentSpaceType.MIXED: partial(
                MixedEncoderNetwork,
                n_continuous_vars=self.n_cont_vars,
                n_discrete_vars=self.n_discrete_vars,
                n_discrete_vals=self.n_discrete_vals,
            ),
        }[latent_space_type]()
        encoder.train(self.training)
        encoder.to(self.device)
        return encoder
        # @endcond

    def get_decoder_network(self, latent_space_type: LatentSpaceType) -> nn.Module:
        """!
        Retrieve the decoder network requested as parameters.
        @param latent_space_type: the type of latent spaces to use for the decoder network
        @return the decoder network
        """
        # @cond IGNORED_BY_DOXYGEN
        decoder = {
            LatentSpaceType.CONTINUOUS: partial(
                ContinuousDecoderNetwork, n_continuous_vars=self.n_cont_vars
            ),
            LatentSpaceType.DISCRETE: partial(
                DiscreteDecoderNetwork,
                n_discrete_vars=self.n_discrete_vars,
                n_discrete_vals=self.n_discrete_vals,
            ),
            LatentSpaceType.MIXED: partial(
                MixedDecoderNetwork,
                n_continuous_vars=self.n_cont_vars,
                n_discrete_vars=self.n_discrete_vars,
                n_discrete_vals=self.n_discrete_vals,
            ),
        }[latent_space_type]()
        decoder.train(self.training)
        decoder.to(self.device)
        return decoder
        # @endcond

    def get_transition_network(self, latent_space_type: LatentSpaceType) -> nn.Module:
        """!
        Retrieve the transition network requested as parameters.
        @param latent_space_type: the type of latent spaces to use for the transition network
        @return the transition network
        """
        # @cond IGNORED_BY_DOXYGEN
        transition = {
            LatentSpaceType.CONTINUOUS: partial(
                ContinuousTransitionNetwork,
                n_actions=self.n_actions,
                n_continuous_vars=self.n_cont_vars,
            ),
            LatentSpaceType.DISCRETE: partial(
                DiscreteTransitionNetwork,
                n_actions=self.n_actions,
                n_discrete_vars=self.n_discrete_vars,
                n_discrete_vals=self.n_discrete_vals,
            ),
            LatentSpaceType.MIXED: partial(
                MixedTransitionNetwork,
                n_actions=self.n_actions,
                n_continuous_vars=self.n_cont_vars,
                n_discrete_vars=self.n_discrete_vars,
                n_discrete_vals=self.n_discrete_vals,
            ),
        }[latent_space_type]()
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
        self, checkpoint_name: str = "", buffer_checkpoint_name: str = ""
    ) -> Tuple[str, Checkpoint]:
        """!
        Load an agent from the filesystem.
        @param checkpoint_name: the name of the agent checkpoint to load
        @param buffer_checkpoint_name: the name of the replay buffer checkpoint to load (None for default name)
        @return a tuple containing the checkpoint path and the checkpoint object
        """

        # Call the parent load function.
        checkpoint_path, checkpoint = super().load(
            checkpoint_name, buffer_checkpoint_name
        )

        # Load the class attributes from the checkpoint.
        self.buffer_size = safe_load(checkpoint, "buffer_size")
        self.batch_size = safe_load(checkpoint, "batch_size")
        self.learning_starts = safe_load(checkpoint, "learning_starts")
        self.n_actions = safe_load(checkpoint, "n_actions")
        self.n_steps = safe_load(checkpoint, "n_steps")
        self.omega = safe_load(checkpoint, "omega")
        self.omega_is = safe_load(checkpoint, "omega_is")
        self.replay_type = safe_load(checkpoint, "replay_type")
        self.learning_rate = safe_load(checkpoint, "learning_rate")
        self.adam_eps = safe_load(checkpoint, "adam_eps")
        self.likelihood_type = safe_load(checkpoint, "likelihood_type")
        self.likelihood_loss = self.get_likelihood_loss(self.likelihood_type)
        self.latent_type = safe_load(checkpoint, "latent_type")
        self.model_loss = self.get_model_loss(self.latent_type)
        self.beta_schedule = safe_load(checkpoint, "beta_schedule")
        self.beta = PiecewiseLinearSchedule(self.beta_schedule)
        self.tau_schedule = safe_load(checkpoint, "tau_schedule")
        self.tau = ExponentialSchedule(self.tau_schedule)
        self.n_cont_vars = safe_load(checkpoint, "n_cont_vars")
        self.n_discrete_vars = safe_load(checkpoint, "n_discrete_vars")
        self.n_discrete_vals = safe_load(checkpoint, "n_discrete_vals")

        # Update the replay buffer.
        replay_buffer = self.get_replay_buffer(
            self.replay_type, self.omega, self.omega_is, self.n_steps
        )
        self.buffer = (
            replay_buffer(capacity=self.buffer_size, batch_size=self.batch_size)
            if self.training
            else None
        )
        self.buffer.load(checkpoint_path, buffer_checkpoint_name)

        # Get the reparameterization function to use with the world model.
        self.reparam = self.get_reparameterization(self.latent_type)
        return checkpoint_path, checkpoint

    def as_dict(self):
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
            "latent_type": self.latent_type,
            "beta_schedule": self.beta_schedule,
            "tau_schedule": self.tau_schedule,
            "n_cont_vars": self.n_cont_vars,
            "n_discrete_vars": self.n_discrete_vars,
            "n_discrete_vals": self.n_discrete_vals,
        } | super().as_dict()
