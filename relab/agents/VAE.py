import logging
import os
from os.path import join
import matplotlib.pyplot as plt

import torch
from torch import optim

from relab.agents.AgentInterface import ReplayType
from relab.agents.VariationalModel import VariationalModel, LikelihoodType, LatentSpaceType
from relab.agents.schedule.ExponentialSchedule import ExponentialSchedule
from relab.agents.schedule.PiecewiseLinearSchedule import PiecewiseLinearSchedule

from relab.helpers.FileSystem import FileSystem
from relab.helpers.MatPlotLib import MatPlotLib
from relab.helpers.VariationalInference import VariationalInference


class VAE(VariationalModel):
    """!
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
        n_continuous_vars=15, n_discrete_vars=20, n_discrete_vals=10,
        learning_rate=0.0001, adam_eps=1e-8, beta_schedule=None, tau_schedule=None,
        replay_type=ReplayType.PRIORITIZED, buffer_size=1000000, batch_size=32, n_steps=1, omega=1.0, omega_is=1.0
    ):
        """!
        Create a Variational Auto-Encoder agent taking random actions.
        @param learning_starts: the step at which learning starts
        @param n_actions: the number of actions available to the agent
        @param training: True if the agent is being trained, False otherwise
        @param likelihood_type: the type of likelihood used by the world model
        @param latent_space_type: the type of latent space used by the world model
        @param n_continuous_vars: the number of continuous latent variables
        @param n_discrete_vars: the number of discrete latent variables
        @param n_discrete_vals: the number of values taken by all the discrete latent variables,
            or a list describing the number of values taken by each discrete latent variable
        @param learning_rate: the learning rate
        @param adam_eps: the epsilon parameter of the Adam optimizer
        @param beta_schedule: the piecewise linear schedule of the KL-divergence weight of beta-VAE
        @param tau_schedule: the exponential schedule of the temperature of the Gumbel-softmax
        @param replay_type: the type of replay buffer
        @param buffer_size: the size of the replay buffer
        @param batch_size: the size of the batches sampled from the replay buffer
        @param n_steps: the number of steps for which rewards are accumulated in multistep Q-learning
        @param omega: the prioritization exponent
        @param omega_is: the important sampling exponent
        """

        # Call the parent constructor.
        super().__init__(
            learning_starts=learning_starts, n_actions=n_actions, training=training,  replay_type=replay_type,
            likelihood_type=likelihood_type, latent_space_type=latent_space_type,
            buffer_size=buffer_size, batch_size=batch_size, n_steps=n_steps, omega=omega, omega_is=omega_is,
            n_continuous_vars=n_continuous_vars, n_discrete_vars=n_discrete_vars, n_discrete_vals=n_discrete_vals,
            learning_rate=learning_rate, adam_eps=adam_eps, beta_schedule=beta_schedule, tau_schedule=tau_schedule,
        )

        # Create the world model.
        self.encoder = self.get_encoder_network(self.latent_space_type)
        self.encoder.train(training)
        self.encoder.to(self.device)
        self.decoder = self.get_decoder_network(self.latent_space_type)
        self.decoder.train(training)
        self.decoder.to(self.device)

        # Create the optimizer.
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.learning_rate, eps=self.adam_eps
        )

    def learn(self):
        """!
        Perform one step of gradient descent on the world model.
        @return the loss of the sampled batch
        """

        # Sample the replay buffer.
        obs, actions, _, _, next_obs = self.buffer.sample()

        # Compute the model loss.
        loss, log_likelihood, kl_divergence = self.model_loss(obs, actions, next_obs)

        # Report the loss obtained for each sampled transition for potential prioritization.
        loss = self.buffer.report(loss)
        loss = loss.mean()

        # Perform one step of gradient descent on the encoder and decoder networks with gradient clipping.
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.encoder.parameters():
            param.grad.data.clamp_(-1, 1)
        for param in self.decoder.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return {
            "vfe": loss,
            "beta": self.beta(self.current_step),
            "log_likelihood": log_likelihood.mean(),
            "kl_divergence": kl_divergence.mean(),
        }

    def continuous_vfe(self, obs, actions, next_obs):
        """!
        Compute the variational free energy for a continuous latent space.
        @param obs: the observations at time t
        @param actions: the actions at time t (unused)
        @param next_obs: the observations at time t + 1 (unused)
        @return the variational free energy
        """

        obs = obs[:, -1, :, :].unsqueeze(dim=1)  # TODO

        # Compute required tensors.
        mean_hat, log_var_hat = self.encoder(obs)
        state = self.reparameterize((mean_hat, log_var_hat))
        reconstructed_obs = self.decoder(state)

        # Compute the variational free energy.
        kl_div_hs = VariationalInference.gaussian_kl_divergence(mean_hat, log_var_hat)
        log_likelihood = self.likelihood_loss(obs, reconstructed_obs)
        vfe_loss = self.beta(self.current_step) * kl_div_hs - log_likelihood
        return vfe_loss, log_likelihood, kl_div_hs

    def discrete_vfe(self, obs, actions, next_obs):
        """!
        Compute the variational free energy for a discrete latent space.
        @param obs: the observations at time t
        @param actions: the actions at time t (unused)
        @param next_obs: the observations at time t + 1 (unused)
        @return the variational free energy
        """

        # Compute required tensors.
        tau = self.tau(self.current_step)
        log_alpha_hats = self.encoder(obs)
        states = self.reparameterize(log_alpha_hats, tau)
        reconstructed_obs = self.decoder(states)

        # Compute the variational free energy.
        kl_div_hs = 0
        for log_alpha_hat in log_alpha_hats:
            kl_div_hs += VariationalInference.categorical_kl_divergence(log_alpha_hat)
        log_likelihood = self.likelihood_loss(obs, reconstructed_obs)
        vfe_loss = self.beta(self.current_step) * kl_div_hs - log_likelihood
        return vfe_loss, log_likelihood, kl_div_hs

    def mixed_vfe(self, obs, actions, next_obs):
        """!
        Compute the variational free energy for a mixed latent space.
        @param obs: the observations at time t
        @param actions: the actions at time t (unused)
        @param next_obs: the observations at time t + 1 (unused)
        @return the variational free energy
        """

        # Compute required tensors.
        tau = self.tau(self.current_step)
        (mean_hat, log_var_hat), log_alpha_hats = self.encoder(obs)
        states = self.reparameterize((mean_hat, log_var_hat), log_alpha_hats, tau)
        reconstructed_obs = self.decoder(states)

        # Compute the variational free energy.
        kl_div_hs = VariationalInference.gaussian_kl_divergence(mean_hat, log_var_hat)
        for log_alpha_hat in log_alpha_hats:
            kl_div_hs += VariationalInference.categorical_kl_divergence(log_alpha_hat)
        log_likelihood = self.likelihood_loss(obs, reconstructed_obs)
        vfe_loss = self.beta(self.current_step) * kl_div_hs - log_likelihood
        return vfe_loss, log_likelihood, kl_div_hs

    def draw_reconstructed_images(self, env, model_index, grid_size):
        """!
        Draw the ground truth and reconstructed images.
        @param env: the gym environment
        @param model_index: the index of the model for which images are generated
        @param grid_size: the size of the image grid to generate
        @return the figure containing the images
        """

        # Create the figure and the grid specification.
        height, width = grid_size
        n_cols = 2
        fig = plt.figure(figsize=(width + n_cols, height * 2))
        gs = fig.add_gridspec(height * 2, width + n_cols)

        # Iterate over the grid's rows.
        tau = self.tau(model_index)
        for h in range(height):

            # Draw the ground truth label for each row.
            fig.add_subplot(gs[2 * h, 0:3])
            plt.text(0.08, 0.45, "Ground Truth Image:", fontsize=10)
            plt.axis("off")

            # Draw the reconstructed image label for each row.
            fig.add_subplot(gs[2 * h + 1, 0:3])
            plt.text(0.08, 0.45, "Reconstructed Image:", fontsize=10)
            plt.axis("off")

            # Iterate over the grid's columns.
            for w in range(width):

                # Retrieve the ground truth and reconstructed images.
                obs, _ = env.reset()
                obs = torch.unsqueeze(obs, dim=0).to(self.device)
                obs = obs[:, -1, :, :].unsqueeze(dim=1)  # TODO
                parameters = self.encoder(obs)
                state = self.reparameterize(parameters, tau=tau)
                reconstructed_obs = self.reconstructed_image_from(self.decoder(state))
                obs = obs[:, -1, :, :].repeat(1, 3, 1, 1)  # TODO
                reconstructed_obs = reconstructed_obs[:, -1, :, :].repeat(1, 3, 1, 1)  # TODO

                # Draw the ground truth image.
                fig.add_subplot(gs[2 * h, w + n_cols])
                plt.imshow(MatPlotLib.format_image(obs))
                plt.axis("off")

                # Draw the reconstructed image.
                fig.add_subplot(gs[2 * h + 1, w + n_cols])
                plt.imshow(MatPlotLib.format_image(reconstructed_obs))
                plt.axis("off")

        # Set spacing between subplots.
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout(pad=0.1)
        return fig

    def load(self, checkpoint_name=None, buffer_checkpoint_name=None):
        """!
        Load an agent from the filesystem.
        @param checkpoint_name: the name of the agent checkpoint to load
        @param buffer_checkpoint_name: the name of the replay buffer checkpoint to load (None for default name)
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
        self.learning_rate = self.safe_load(checkpoint, "learning_rate")
        self.adam_eps = self.safe_load(checkpoint, "adam_eps")
        self.likelihood_type = self.safe_load(checkpoint, "likelihood_type")
        self.latent_space_type = self.safe_load(checkpoint, "latent_space_type")
        self.n_continuous_vars = self.safe_load(checkpoint, "n_continuous_vars")
        self.n_discrete_vars = self.safe_load(checkpoint, "n_discrete_vars")
        self.n_discrete_vals = self.safe_load(checkpoint, "n_discrete_vals")
        self.beta_schedule = self.safe_load(checkpoint, "beta_schedule")
        self.beta = PiecewiseLinearSchedule(self.beta_schedule)
        self.tau_schedule = self.safe_load(checkpoint, "tau_schedule")
        self.tau = ExponentialSchedule(self.tau_schedule)

        # Update the model loss using the checkpoint.
        self.model_loss = self.get_model_loss(self.latent_space_type)
        self.likelihood_loss = self.get_likelihood_loss(self.likelihood_type)

        # Update the world model using the checkpoint.
        self.encoder = self.get_encoder_network(self.latent_space_type)
        self.encoder.load_state_dict(self.safe_load(checkpoint, "encoder"))
        self.encoder.train(self.training)
        self.encoder.to(self.device)
        self.decoder = self.get_decoder_network(self.latent_space_type)
        self.decoder.load_state_dict(self.safe_load(checkpoint, "decoder"))
        self.decoder.train(self.training)
        self.decoder.to(self.device)

        # Update the replay buffer.
        replay_buffer = self.get_replay_buffer(self.replay_type, self.omega, self.omega_is, self.n_steps)
        self.buffer = replay_buffer(capacity=self.buffer_size, batch_size=self.batch_size) if self.training else None
        self.buffer.load(checkpoint_path, buffer_checkpoint_name)

        # Get the reparameterization function to use with the world model.
        self.reparameterize = self.get_reparameterization(self.latent_space_type)

        # Update the optimizer.
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.learning_rate, eps=self.adam_eps
        )
        self.optimizer.load_state_dict(self.safe_load(checkpoint, "optimizer"))

    def save(self, checkpoint_name, buffer_checkpoint_name=None):
        """!
        Save the agent on the filesystem.
        @param checkpoint_name: the name of the checkpoint in which to save the agent
        @param buffer_checkpoint_name: the name of the checkpoint to save the replay buffer (None for default name)
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
            "n_continuous_vars": self.n_continuous_vars,
            "n_discrete_vars": self.n_discrete_vars,
            "n_discrete_vals": self.n_discrete_vals,
            "beta_schedule": self.beta_schedule,
            "tau_schedule": self.tau_schedule,
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }, checkpoint_path)

        # Save the replay buffer.
        self.buffer.save(checkpoint_path, buffer_checkpoint_name)
