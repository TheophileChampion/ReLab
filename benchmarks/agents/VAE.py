import logging
import os
from functools import partial
from os.path import join

import torch
from torch import optim

from benchmarks.agents.AgentInterface import ReplayType
from benchmarks.agents.Random import Random, LikelihoodType, LatentSpaceType
from benchmarks.agents.networks.DecoderNetworks import ContinuousDecoderNetwork, DiscreteDecoderNetwork, \
    MixedDecoderNetwork
from benchmarks.agents.networks.EncoderNetworks import ContinuousEncoderNetwork, DiscreteEncoderNetwork, \
    MixedEncoderNetwork
from benchmarks.agents.schedule.ExponentialSchedule import ExponentialSchedule
from benchmarks.agents.schedule.PiecewiseLinearSchedule import PiecewiseLinearSchedule

from benchmarks.helpers.FileSystem import FileSystem
from benchmarks.helpers.VariationalInference import VariationalInference


class VAE(Random):
    """
    Implement an agent taking random actions, and learning a world model using a Variational Auto-Encoder (VAE) from:
    Kingma Diederi, and Welling Max. Auto-Encoding Variational Bayes.
    International Conference on Learning Representations, 2014.

    This implementation also support beta-VAE [1], Concrete VAE [2], Joint VAE [3], and prioritized replay buffer [4]:
    [1] Irina Higgins, Loic Matthey, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick,
        Shakir Mohamed, and Alexander Lerchner.
        beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. ICLR, 2017.
    [2] Eric Jang, Shixiang Gu, and Ben Poole. Categorical Reparameterization with Gumbel-Softmax.
        arXiv preprint arXiv:1611.01144, 2016.
    [3] Emilien Dupont. Learning Disentangled Joint Continuous and Discrete Representations. NeurIPS, 2018.
    [4] Tom Schaul. Prioritized experience replay. arXiv preprint arXiv:1511.05952, 2015.
    """

    def __init__(
        self, learning_starts=200000, n_actions=18, training=True,
        likelihood_type=LikelihoodType.BERNOULLI, latent_space_type=LatentSpaceType.CONTINUOUS,
        n_continuous_vars=10, n_discrete_vars=20, n_discrete_vals=10,
        learning_rate=0.00001, adam_eps=1.5e-4, beta_schedule=None, tau_schedule=None,
        replay_type=ReplayType.DEFAULT, buffer_size=1000000, batch_size=32, n_steps=1, omega=1.0, omega_is=1.0
    ):
        """
        Create a Variational Auto-Encoder agent taking random actions.
        :param learning_starts: the step at which learning starts
        :param n_actions: the number of actions available to the agent
        :param training: True if the agent is being trained, False otherwise
        :param likelihood_type: the type of likelihood used by the world model
        :param latent_space_type: the type of latent space used by the world model
        :param n_continuous_vars: the number of continuous latent variables
        :param n_discrete_vars: the number of discrete latent variables
        :param n_discrete_vals: the number of values taken by all the discrete latent variables,
            or a list describing the number of values taken by each discrete latent variable
        :param learning_rate: the learning rate
        :param adam_eps: the epsilon parameter of the Adam optimizer
        :param beta_schedule: the piecewise linear schedule of the KL-divergence weight of beta-VAE
        :param tau_schedule: the exponential schedule of the temperature of the Gumbel-softmax
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

        # Store the agent's parameters.
        self.likelihood_type = likelihood_type
        self.latent_space_type = latent_space_type
        self.learning_rate = learning_rate
        self.adam_eps = adam_eps
        self.n_continuous_vars = n_continuous_vars
        self.n_discrete_vars = n_discrete_vars
        self.n_discrete_vals = n_discrete_vals
        self.beta_schedule = [(1.0, 0)] if beta_schedule is None else beta_schedule
        self.beta = PiecewiseLinearSchedule(self.beta_schedule)
        self.tau_schedule = (0.5, -3e-5) if tau_schedule is None else tau_schedule
        self.tau = ExponentialSchedule(self.tau_schedule)

        # Get the loss used by the world model.
        self.model_loss = self.get_model_loss(self.latent_space_type)
        self.likelihood_loss = self.get_likelihood_loss(self.likelihood_type)

        # Create the world model.
        self.encoder = self.get_encoder_network(self.latent_space_type)
        self.decoder = self.get_decoder_network(self.latent_space_type)

        # Create the optimizer.
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.learning_rate, eps=self.adam_eps
        )

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

    def learn(self):
        """
        Perform one step of gradient descent on the world model.
        """

        # Sample the replay buffer.
        obs, _, _, _, _ = self.buffer.sample()

        # Compute the model loss.
        loss = self.model_loss(obs)

        # Report the loss obtained for each sampled transition for potential prioritization.
        loss = self.buffer.report(loss)

        # Perform one step of gradient descent on the encoder and decoder networks with gradient clipping.
        self.optimizer.zero_grad()
        loss.mean().backward()
        for param in self.encoder.parameters():
            param.grad.data.clamp_(-1, 1)
        for param in self.decoder.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def continuous_vfe(self, obs):
        """
        Compute the variational free energy for a continuous latent space.
        :param obs: the observations at time t
        :return: the variational free energy
        """

        # Compute required tensors.
        mean_hat, log_var_hat = self.encoder(obs)
        state = VariationalInference.gaussian_reparameterization(mean_hat, log_var_hat)
        reconstructed_obs = self.decoder(state)

        # Compute the variational free energy.
        kl_div_hs = VariationalInference.gaussian_kl_divergence(mean_hat, log_var_hat)
        log_likelihood = self.likelihood_loss(obs, reconstructed_obs)
        vfe_loss = self.beta(self.current_step) * kl_div_hs - log_likelihood
        return vfe_loss

    def discrete_vfe(self, obs):
        """
        Compute the variational free energy for a discrete latent space.
        :param obs: the observations at time t
        :return: the variational free energy
        """

        # Compute required tensors.
        log_alpha_hats = self.encoder(obs)
        states = [
            VariationalInference.concrete_reparameterization(log_alpha_hat, self.tau(self.current_step))
            for log_alpha_hat in log_alpha_hats
        ]
        state = torch.cat(states)
        reconstructed_obs = self.decoder(state)

        # Compute the variational free energy.
        kl_div_hs = 0
        for log_alpha_hat in log_alpha_hats:
            kl_div_hs += VariationalInference.categorical_kl_divergence(log_alpha_hat)
        log_likelihood = self.likelihood_loss(obs, reconstructed_obs)
        vfe_loss = self.beta(self.current_step) * kl_div_hs - log_likelihood
        return vfe_loss

    def mixed_vfe(self, obs):
        """
        Compute the variational free energy for a mixed latent space.
        :param obs: the observations at time t
        :return: the variational free energy
        """

        # Compute required tensors.
        (mean_hat, log_var_hat), log_alpha_hats = self.encoder(obs)
        states = [
            VariationalInference.concrete_reparameterization(log_alpha_hat, self.tau(self.current_step))
            for log_alpha_hat in log_alpha_hats
        ]
        states.append(VariationalInference.gaussian_reparameterization(mean_hat, log_var_hat))
        state = torch.cat(states)
        reconstructed_obs = self.decoder(state)

        # Compute the variational free energy.
        kl_div_hs = VariationalInference.gaussian_kl_divergence(mean_hat, log_var_hat)
        for log_alpha_hat in log_alpha_hats:
            kl_div_hs += VariationalInference.categorical_kl_divergence(log_alpha_hat)
        log_likelihood = self.likelihood_loss(obs, reconstructed_obs)
        vfe_loss = self.beta(self.current_step) * kl_div_hs - log_likelihood
        return vfe_loss

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
        self.learning_rate = self.safe_load(checkpoint, "learning_rate")
        self.adam_eps = self.safe_load(checkpoint, "adam_eps")
        self.likelihood_type = self.safe_load(checkpoint, "likelihood_type")
        self.latent_space_type = self.safe_load(checkpoint, "latent_space_type")
        self.beta_schedule = self.safe_load(checkpoint, "beta_schedule")
        self.beta = PiecewiseLinearSchedule(self.beta_schedule)
        self.tau_schedule = self.safe_load(checkpoint, "tau_schedule")
        self.tau = ExponentialSchedule(self.tau_schedule)

        # Update the model loss using the checkpoint.
        self.model_loss = self.get_model_loss(self.latent_space_type)
        self.likelihood_loss = self.get_likelihood_loss(self.likelihood_type)

        # Update the world model using the checkpoint.
        self.encoder = self.get_encoder_network(self.latent_space_type)
        self.decoder = self.get_decoder_network(self.latent_space_type)

        # Update the replay buffer.
        replay_buffer = self.get_replay_buffer(self.replay_type, self.omega, self.omega_is, self.n_steps)
        self.buffer = replay_buffer(capacity=self.buffer_size, batch_size=self.batch_size) if self.training else None

        # Update the optimizer.
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.learning_rate, eps=self.adam_eps
        )

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
            "learning_rate": self.learning_rate,
            "adam_eps": self.adam_eps,
            "likelihood_type": self.likelihood_type,
            "latent_space_type": self.latent_space_type,
            "beta_schedule": self.beta_schedule,
            "tau_schedule": self.tau_schedule
        }, checkpoint_path)
