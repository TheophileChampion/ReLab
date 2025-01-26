import logging
import os
from os.path import join
from typing import Any, Dict, Tuple, Optional
from gymnasium import Env
from matplotlib.figure import Figure

import matplotlib.pyplot as plt

import torch
from torch import optim, Tensor

from relab.agents.AgentInterface import ReplayType
from relab.agents.VariationalModel import VariationalModel, LikelihoodType, LatentSpaceType

from relab.helpers.FileSystem import FileSystem
from relab.helpers.MatPlotLib import MatPlotLib
from relab.helpers.Typing import Checkpoint
from relab.helpers.VariationalInference import VariationalInference


class HMM(VariationalModel):
    """!
    @brief Implements a Hidden Markov Model.

    @details
    This implementation extends upon the paper:

    <b>Auto-Encoding Variational Bayes</b>,
    published in ICLR, 2014.

    Authors:
    - Kingma D.
    - Welling M.

    More precisely, the HMM extends the VAE framework to sequential data
    by modeling temporal dependencies using a transition network.
    Note, this agent takes random actions.
    """

    def __init__(
        self,
        learning_starts : int = 200000,
        n_actions : int = 18,
        training : bool = True,
        likelihood_type : LikelihoodType = LikelihoodType.BERNOULLI,
        latent_space_type : LatentSpaceType = LatentSpaceType.CONTINUOUS,
        n_continuous_vars : int = 10,
        n_discrete_vars : int = 20,
        n_discrete_vals : int = 10,
        learning_rate : float = 0.00001,
        adam_eps : float = 1.5e-4,
        beta_schedule : Any = None,
        tau_schedule : Any = None,
        replay_type : ReplayType = ReplayType.DEFAULT,
        buffer_size : int = 1000000,
        batch_size : int = 32,
        n_steps : int = 1,
        omega : float = 1.0,
        omega_is : float = 1.0
    ) -> None:
        """!
        Create a Hidden Markov Model agent taking random actions.
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

        ## @var encoder
        # Neural network that encodes observations into latent states.
        self.encoder = self.get_encoder_network(self.latent_space_type)

        ## @var decoder
        # Neural network that decodes latent states into observations.
        self.decoder = self.get_decoder_network(self.latent_space_type)

        ## @var transition
        # Neural network that models the transition dynamics in latent space.
        self.transition = self.get_transition_network(self.latent_space_type)

        ## @var optimizer
        # Adam optimizer for training the encoder, decoder, and transition networks.
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.transition.parameters()),
            lr=self.learning_rate, eps=self.adam_eps
        )

    def learn(self) -> Optional[Dict[str, Any]]:
        """!
        Perform one step of gradient descent on the world model.
        @return the loss of the sampled batch
        """
        # @cond IGNORED_BY_DOXYGEN

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
        # @endcond

    def continuous_vfe(self, obs : Tensor, actions : Tensor, next_obs : Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """!
        Compute the variational free energy for a continuous latent space.
        @param obs: the observations at time t
        @param actions: the actions at time t
        @param next_obs: the observations at time t + 1
        @return the variational free energy
        """

        # Compute required vectors.
        mean_hat, log_var_hat = self.encoder(obs)
        states = self.reparameterize((mean_hat, log_var_hat))
        reconstructed_obs = self.decoder(states)
        next_mean_hat, next_log_var_hat = self.encoder(next_obs)
        next_states = self.reparameterize((next_mean_hat, next_log_var_hat))
        next_reconstructed_obs = self.decoder(next_states)
        mean, log_var = self.transition(states, actions)

        # Compute the variational free energy.
        kl_div_hs = VariationalInference.gaussian_kl_divergence(mean_hat, log_var_hat) + \
            VariationalInference.gaussian_kl_divergence(next_mean_hat, next_log_var_hat, mean, log_var)
        log_likelihood = self.likelihood_loss(obs, reconstructed_obs) + \
            self.likelihood_loss(next_obs, next_reconstructed_obs)
        vfe_loss = self.beta(self.current_step) * kl_div_hs - log_likelihood
        return vfe_loss, log_likelihood, kl_div_hs

    def discrete_vfe(self, obs : Tensor, actions : Tensor, next_obs : Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """!
        Compute the variational free energy for a discrete latent space.
        @param obs: the observations at time t
        @param actions: the actions at time t
        @param next_obs: the observations at time t + 1
        @return the variational free energy
        """
        # @cond IGNORED_BY_DOXYGEN

        # Compute required vectors.
        tau = self.tau(self.current_step)
        log_alpha_hats = self.encoder(obs)
        states = self.reparameterize(log_alpha_hats, tau=tau)
        reconstructed_obs = self.decoder(states)
        next_log_alpha_hats = self.encoder(next_obs)
        next_states = self.reparameterize(next_log_alpha_hats, tau=tau)
        next_reconstructed_obs = self.decoder(next_states)
        log_alphas = self.transition(states, actions)

        # Compute the variational free energy.
        kl_div_hs_t0 = 0
        for log_alpha_hat in log_alpha_hats:
            kl_div_hs_t0 += VariationalInference.categorical_kl_divergence(log_alpha_hat)
        kl_div_hs_t1 = 0
        for next_log_alpha_hat, log_alpha in zip(next_log_alpha_hats, log_alphas):
            kl_div_hs_t1 += VariationalInference.categorical_kl_divergence(next_log_alpha_hat, log_alpha)
        kl_div_hs = kl_div_hs_t1 + kl_div_hs_t0
        log_likelihood = self.likelihood_loss(obs, reconstructed_obs) + \
            self.likelihood_loss(next_obs, next_reconstructed_obs)
        vfe_loss = self.beta(self.current_step) * kl_div_hs - log_likelihood
        return vfe_loss, log_likelihood, kl_div_hs
        # @endcond

    def mixed_vfe(self, obs : Tensor, actions : Tensor, next_obs : Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """!
        Compute the variational free energy for a mixed latent space.
        @param obs: the observations at time t
        @param actions: the actions at time t
        @param next_obs: the observations at time t + 1
        @return the variational free energy
        """
        # @cond IGNORED_BY_DOXYGEN

        # Compute required vectors.
        tau = self.tau(self.current_step)
        (mean_hat, log_var_hat), log_alpha_hats = self.encoder(obs)
        states = self.reparameterize((mean_hat, log_var_hat), log_alpha_hats, tau=tau)
        reconstructed_obs = self.decoder(states)
        (next_mean_hat, next_log_var_hat), next_log_alpha_hats = self.encoder(next_obs)
        next_states = self.reparameterize((next_mean_hat, next_log_var_hat), next_log_alpha_hats, tau=tau)
        next_reconstructed_obs = self.decoder(next_states)
        (mean, log_var), log_alphas = self.transition(states, actions)

        # Compute the variational free energy.
        kl_div_hs_t0 = VariationalInference.gaussian_kl_divergence(mean_hat, log_var_hat)
        for log_alpha_hat in log_alpha_hats:
            kl_div_hs_t0 += VariationalInference.categorical_kl_divergence(log_alpha_hat)
        kl_div_hs_t1 = VariationalInference.gaussian_kl_divergence(next_mean_hat, next_log_var_hat, mean, log_var)
        for next_log_alpha_hat, log_alpha in zip(next_log_alpha_hats, log_alphas):
            kl_div_hs_t1 += VariationalInference.categorical_kl_divergence(next_log_alpha_hat, log_alpha)
        kl_div_hs = kl_div_hs_t1 + kl_div_hs_t0
        log_likelihood = self.likelihood_loss(obs, reconstructed_obs) + \
            self.likelihood_loss(next_obs, next_reconstructed_obs)
        vfe_loss = self.beta(self.current_step) * kl_div_hs - log_likelihood
        return vfe_loss, log_likelihood, kl_div_hs
        # @endcond

    def draw_reconstructed_images(self, env : Env, model_index : int , grid_size : Tuple[int, int]) -> Figure:
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
        h = 0
        tau = self.tau(model_index)
        while h < height:

            # Draw the ground truth label for each row.
            fig.add_subplot(gs[2 * h, 0:3])
            plt.text(0.08, 0.45, "Ground Truth Image:", fontsize=10)
            plt.axis("off")

            # Draw the reconstructed image label for each row.
            fig.add_subplot(gs[2 * h + 1, 0:3])
            plt.text(0.08, 0.45, "Reconstructed Image:", fontsize=10)
            plt.axis("off")

            # Retrieve the initial ground truth and reconstructed images.
            obs, _ = env.reset()
            obs = torch.unsqueeze(obs, dim=0)
            parameters = self.encoder(obs)
            states = self.reparameterize(parameters, tau=tau)
            reconstructed_obs = self.reconstructed_image_from(self.decoder(states))

            # Iterate over the grid's columns.
            for w in range(width):

                # Draw the ground truth image.
                fig.add_subplot(gs[2 * h, w + n_cols])
                plt.imshow(MatPlotLib.format_image(obs))
                plt.axis("off")

                # Draw the reconstructed image.
                fig.add_subplot(gs[2 * h + 1, w + n_cols])
                plt.imshow(MatPlotLib.format_image(reconstructed_obs))
                plt.axis("off")

                # Execute the agent's action in the environment to obtain the next ground truth observation.
                action = self.step(obs)
                obs, _, terminated, truncated, _ = env.step(action)
                obs = torch.unsqueeze(obs, dim=0)
                done = terminated or truncated
                action = torch.tensor([action])

                # Simulate the agent's action to obtain the next reconstructed observation.
                parameters = self.transition(states, action)
                states = self.reparameterize(parameters, tau=tau)
                reconstructed_obs = torch.sigmoid(self.decoder(states))

                # Increase row index.
                if done:
                    h -= 1
                    break

            # Increase row index.
            h += 1

        # Set spacing between subplots.
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout(pad=0.1)
        return fig

    def load(
        self,
        checkpoint_name : Optional[str] = None,
        buffer_checkpoint_name : Optional[str] = None
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
            checkpoint_path, checkpoint = super().load(checkpoint_name, buffer_checkpoint_name)

            # Update the world model using the checkpoint.
            self.encoder = self.get_encoder_network(self.latent_space_type)
            self.encoder.load_state_dict(self.safe_load(checkpoint, "encoder"))
            self.encoder.train(self.training)
            self.encoder.to(self.device)
            self.decoder = self.get_decoder_network(self.latent_space_type)
            self.decoder.load_state_dict(self.safe_load(checkpoint, "decoder"))
            self.decoder.train(self.training)
            self.decoder.to(self.device)
            self.transition = self.get_transition_network(self.latent_space_type)
            self.transition.load_state_dict(self.safe_load(checkpoint, "transition"))
            self.transition.train(self.training)
            self.transition.to(self.device)

            # Update the replay buffer.
            # TODO move to VariationalModel.load?
            replay_buffer = self.get_replay_buffer(self.replay_type, self.omega, self.omega_is, self.n_steps)
            self.buffer = replay_buffer(capacity=self.buffer_size, batch_size=self.batch_size) if self.training else None
            self.buffer.load(checkpoint_path, buffer_checkpoint_name)

            # Update the reparameterization function to use with the world model.
            self.reparameterize = self.get_reparameterization(self.latent_space_type)

            # Update the optimizer.
            params = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.transition.parameters())
            self.optimizer = self.safe_load_optimizer(checkpoint, params, self.learning_rate, self.adam_eps)
            return checkpoint_path, checkpoint

        # Catch the exception raise if the checkpoint could not be located.
        except FileNotFoundError:
            return "", None
        # @endcond

    def as_dict(self):
        """"
        Convert the agent into a dictionary that can be saved on the filesystem.
        @return the dictionary
        """
        return {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "transition": self.transition.state_dict(),
            "optimizer": self.optimizer.state_dict()
        } | super().as_dict()

    def save(self, checkpoint_name : str, buffer_checkpoint_name : Optional[str] = None) -> None:
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
        torch.save(self.as_dict(), checkpoint_path)

        # Save the replay buffer.
        self.buffer.save(checkpoint_path, buffer_checkpoint_name)
        # @endcond
