from abc import ABC
import abc
import logging
import os
from datetime import datetime
from os.path import join
from enum import IntEnum
from functools import partial
import re

import torch

import benchmarks
from benchmarks.agents.AgentInterface import AgentInterface, ReplayType
from benchmarks.agents.memory.ReplayBuffer import Experience
import numpy as np
from benchmarks.agents.networks.DecoderNetworks import ContinuousDecoderNetwork, DiscreteDecoderNetwork, \
    MixedDecoderNetwork
from benchmarks.agents.networks.EncoderNetworks import ContinuousEncoderNetwork, DiscreteEncoderNetwork, \
    MixedEncoderNetwork
from benchmarks.agents.networks.TransitionNetworks import ContinuousTransitionNetwork, DiscreteTransitionNetwork, \
    MixedTransitionNetwork

from benchmarks.helpers.VariationalInference import VariationalInference
from benchmarks.helpers.MatPlotLib import MatPlotLib
from benchmarks.agents.schedule.ExponentialSchedule import ExponentialSchedule
from benchmarks.agents.schedule.PiecewiseLinearSchedule import PiecewiseLinearSchedule


class LatentSpaceType(IntEnum):
    """
    The type of latent spaces supported by the model-based agents.
    """
    CONTINUOUS = 0  # Latent space with a continuous latent variables
    DISCRETE = 1  # Latent space with discrete latent variables
    MIXED = 2  # Latent space with discrete and continuous latent variables


class LikelihoodType(IntEnum):
    """
    The type of likelihood supported by the model-based agents.
    """
    GAUSSIAN = 0  # Gaussian likelihood
    BERNOULLI = 1  # Bernoulli likelihood


class VariationalModel(AgentInterface, ABC):
    """
    Implement an agent taking random actions, with support for learning world models.
    """

    def __init__(
        self, learning_starts=200000, n_actions=18, training=False,
        likelihood_type=LikelihoodType.BERNOULLI, latent_space_type=LatentSpaceType.CONTINUOUS,
        replay_type=ReplayType.DEFAULT, buffer_size=1000000, batch_size=32, n_steps=1, omega=1.0, omega_is=1.0,
        n_continuous_vars=10, n_discrete_vars=20, n_discrete_vals=10,
        learning_rate=0.00001, adam_eps=1.5e-4, beta_schedule=None, tau_schedule=None,
    ):
        """
        Create an agent taking random actions.
        :param learning_starts: the step at which learning starts
        :param n_actions: the number of actions available to the agent
        :param training: True if the agent is being trained, False otherwise
        :param likelihood_type: the type of likelihood used by the world model
        :param latent_space_type: the type of latent space used by the world model
        :param replay_type: the type of replay buffer
        :param buffer_size: the size of the replay buffer
        :param batch_size: the size of the batches sampled from the replay buffer
        :param n_steps: the number of steps for which rewards are accumulated in multistep Q-learning
        :param omega: the prioritization exponent
        :param omega_is: the important sampling exponent
        :param n_continuous_vars: the number of continuous latent variables
        :param n_discrete_vars: the number of discrete latent variables
        :param n_discrete_vals: the number of values taken by all the discrete latent variables,
            or a list describing the number of values taken by each discrete latent variable
        :param learning_rate: the learning rate
        :param adam_eps: the epsilon parameter of the Adam optimizer
        :param beta_schedule: the piecewise linear schedule of the KL-divergence weight of beta-VAE
        :param tau_schedule: the exponential schedule of the temperature of the Gumbel-softmax
        """

        # Call the parent constructor.
        super().__init__(training=training)

        # Store the agent's parameters.
        self.likelihood_type = likelihood_type
        self.latent_space_type = latent_space_type
        self.learning_starts = learning_starts
        self.n_actions = n_actions
        self.training = training
        self.replay_type = replay_type
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.n_continuous_vars = n_continuous_vars
        self.n_discrete_vars = n_discrete_vars
        self.n_discrete_vals = n_discrete_vals
        self.n_steps = n_steps
        self.omega = omega
        self.omega_is = omega_is
        self.learning_rate = learning_rate
        self.adam_eps = adam_eps
        self.beta_schedule = [(0, 1.0)] if beta_schedule is None else beta_schedule
        self.beta = PiecewiseLinearSchedule(self.beta_schedule)
        self.tau_schedule = (0.5, -3e-5) if tau_schedule is None else tau_schedule
        self.tau = ExponentialSchedule(self.tau_schedule)

        # Get the losses used by the world model.
        self.model_loss = self.get_model_loss(self.latent_space_type)
        self.likelihood_loss = self.get_likelihood_loss(self.likelihood_type)

        # Get the reparameterization function to use with the world model.
        self.reparameterize = self.get_reparameterization(self.latent_space_type)

        # Create the replay buffer.
        replay_buffer = self.get_replay_buffer(self.replay_type, self.omega, self.omega_is, self.n_steps)
        self.buffer = replay_buffer(capacity=self.buffer_size, batch_size=self.batch_size) if self.training else None

    @abc.abstractmethod
    def learn(self):
        """
        Perform one step of gradient descent on the world model.
        :return: the loss of the sampled batch, None if no loss should be logged in Tensorboard
        """
        ...

    @abc.abstractmethod
    def continuous_vfe(self, obs, actions, next_obs):
        """
        Compute the variational free energy for a continuous latent space.
        :param obs: the observations at time t
        :param actions: the actions at time t
        :param next_obs: the observations at time t + 1
        :return: the variational free energy
        """
        ...

    @abc.abstractmethod
    def discrete_vfe(self, obs, actions, next_obs):
        """
        Compute the variational free energy for a discrete latent space.
        :param obs: the observations at time t
        :param actions: the actions at time t
        :param next_obs: the observations at time t + 1
        :return: the variational free energy
        """
        ...

    @abc.abstractmethod
    def mixed_vfe(self, obs, actions, next_obs):
        """
        Compute the variational free energy for a mixed latent space.
        :param obs: the observations at time t
        :param actions: the actions at time t
        :param next_obs: the observations at time t + 1
        :return: the variational free energy
        """
        ...

    @abc.abstractmethod
    def draw_reconstructed_images(self, env, model_index, grid_size):
        """
        Draw the ground truth and reconstructed images.
        :param env: the gym environment
        :param model_index: the index of the model for which images are generated
        :param grid_size: the size of the image grid to generate
        :return: the figure containing the images
        """
        ...

    def get_model_loss(self, latent_space_type):
        """
        Retrieve the model loss requested as parameters.
        :param latent_space_type: the type of latent spaces used by the world model
        :return: the loss
        """
        return {
            LatentSpaceType.CONTINUOUS: self.continuous_vfe,
            LatentSpaceType.DISCRETE: self.discrete_vfe,
            LatentSpaceType.MIXED: self.mixed_vfe,
        }[latent_space_type]

    @staticmethod
    def get_reparameterization(latent_space_type):
        """
        Retrieve the reparameterization function requested as parameters.
        :param latent_space_type: the type of latent spaces used by the world model
        :return: the reparameterization function
        """
        return {
            LatentSpaceType.CONTINUOUS: VariationalInference.continuous_reparameterization,
            LatentSpaceType.DISCRETE: VariationalInference.discrete_reparameterization,
            LatentSpaceType.MIXED: VariationalInference.mixed_reparameterization,
        }[latent_space_type]

    @staticmethod
    def get_likelihood_loss(likelihood_type):
        """
        Retrieve the likelihood loss requested as parameters.
        :param likelihood_type: the type of likelihood used by the world model
        :return: the loss
        """
        return {
            LikelihoodType.GAUSSIAN: VariationalInference.gaussian_log_likelihood,
            LikelihoodType.BERNOULLI: VariationalInference.bernoulli_log_likelihood,
        }[likelihood_type]

    def get_encoder_network(self, latent_space_type):
        """
        Retrieve the encoder network requested as parameters.
        :param latent_space_type: the type of latent spaces to use for the encoder network
        :return: the encoder network
        """
        encoder = {
            LatentSpaceType.CONTINUOUS: partial(ContinuousEncoderNetwork, n_continuous_vars=self.n_continuous_vars),
            LatentSpaceType.DISCRETE: partial(DiscreteEncoderNetwork, n_discrete_vars=self.n_discrete_vars, n_discrete_vals=self.n_discrete_vals),
            LatentSpaceType.MIXED: partial(MixedEncoderNetwork, n_continuous_vars=self.n_continuous_vars, n_discrete_vars=self.n_discrete_vars, n_discrete_vals=self.n_discrete_vals),
        }[latent_space_type]
        return encoder()

    def get_decoder_network(self, latent_space_type):
        """
        Retrieve the decoder network requested as parameters.
        :param latent_space_type: the type of latent spaces to use for the decoder network
        :return: the decoder network
        """
        decoder = {
            LatentSpaceType.CONTINUOUS: partial(ContinuousDecoderNetwork, n_continuous_vars=self.n_continuous_vars),
            LatentSpaceType.DISCRETE: partial(DiscreteDecoderNetwork, n_discrete_vars=self.n_discrete_vars, n_discrete_vals=self.n_discrete_vals),
            LatentSpaceType.MIXED: partial(MixedDecoderNetwork, n_continuous_vars=self.n_continuous_vars, n_discrete_vars=self.n_discrete_vars, n_discrete_vals=self.n_discrete_vals),
        }[latent_space_type]
        return decoder()

    def get_transition_network(self, latent_space_type):
        """
        Retrieve the transition network requested as parameters.
        :param latent_space_type: the type of latent spaces to use for the transition network
        :return: the transition network
        """
        transition = {
            LatentSpaceType.CONTINUOUS: partial(ContinuousTransitionNetwork, n_actions=self.n_actions, n_continuous_vars=self.n_continuous_vars),
            LatentSpaceType.DISCRETE: partial(DiscreteTransitionNetwork, n_actions=self.n_actions, n_discrete_vars=self.n_discrete_vars, n_discrete_vals=self.n_discrete_vals),
            LatentSpaceType.MIXED: partial(MixedTransitionNetwork, n_actions=self.n_actions, n_continuous_vars=self.n_continuous_vars, n_discrete_vars=self.n_discrete_vars, n_discrete_vals=self.n_discrete_vals),
        }[latent_space_type]
        return transition()

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

    def demo(self, env, gif_name, max_frames=10000):
        """
        Demonstrate the agent policy in the gym environment passed as parameters
        :param env: the gym environment
        :param gif_name: the name of the GIF file in which to save the demo
        :param max_frames: the maximum number of frames to include in the GIF file
        """

        # Create the GIF file containing a demonstration of the agent's policy.
        super().demo(env, gif_name, max_frames)

        # Create a graph containing images generated by the world model.
        model_index = int(re.findall(r"\d+", gif_name)[0])
        fig = self.draw_reconstructed_images(env, model_index, grid_size=(6, 6))

        # Save the figure containing the ground truth and reconstructed images.
        figure_name = gif_name.replace(".gif", "") + "_reconstructed_images.pdf"
        figure_path = join(os.environ["DEMO_DIRECTORY"], figure_name)
        fig.savefig(figure_path)
        MatPlotLib.close()

    def reconstructed_image_from(self, decoder_output):
        """
        Compute the reconstructed image from the decoder output.
        :param decoder_output: the tensor predicted by the decoder
        :return: the reconstructed image
        """
        function = {
            LikelihoodType.GAUSSIAN: partial(torch.clamp, min=0, max=1),
            LikelihoodType.BERNOULLI: torch.sigmoid,
        }[self.likelihood_type]
        return function(decoder_output)
