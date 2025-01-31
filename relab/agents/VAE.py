import logging
from os.path import join
from typing import Any, Dict, Tuple, Optional

import matplotlib.pyplot as plt
from gymnasium import Env
from matplotlib.figure import Figure

import torch
from torch import Tensor

from relab.agents.AgentInterface import ReplayType
from relab.agents.VariationalModel import (
    VariationalModel,
    LikelihoodType,
    LatentSpaceType,
)
from relab.helpers.Serialization import get_optimizer, safe_load_state_dict

from relab.helpers.FileSystem import FileSystem
from relab.helpers.MatPlotLib import MatPlotLib
from relab.helpers.Typing import Checkpoint
from relab.helpers.VariationalInference import (
    gaussian_kl_divergence as gauss_kl,
    sum_categorical_kl_divergences as sum_cat_kl,
)


class VAE(VariationalModel):
    """!
    @brief Implements a Variational Auto-Encoder (VAE) agent.

    @details
    This implementation is based on the paper:

    <b>Auto-Encoding Variational Bayes</b>,
    published in ICLR, 2014.

    Authors:
    - Kingma Diederi
    - Welling Max

    The paper introduced the VAE algorithm, variational inference with deep neural
    networks to unsupervised embedding of images on the Frey Face and MNIST datasets.
    Note, this agent takes random actions.
    """

    def __init__(
        self,
        learning_starts: int = 200000,
        n_actions: int = 18,
        training: bool = True,
        likelihood_type: LikelihoodType = LikelihoodType.BERNOULLI,
        latent_space_type: LatentSpaceType = LatentSpaceType.CONTINUOUS,
        n_continuous_vars: int = 15,
        n_discrete_vars: int = 20,
        n_discrete_vals: int = 10,
        learning_rate: float = 0.0001,
        adam_eps: float = 1e-8,
        beta_schedule: Any = None,
        tau_schedule: Any = None,
        replay_type: ReplayType = ReplayType.PRIORITIZED,
        buffer_size: int = 1000000,
        batch_size: int = 32,
        n_steps: int = 1,
        omega: float = 1.0,
        omega_is: float = 1.0,
    ) -> None:
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
            learning_starts=learning_starts,
            n_actions=n_actions,
            training=training,
            replay_type=replay_type,
            likelihood_type=likelihood_type,
            latent_space_type=latent_space_type,
            buffer_size=buffer_size,
            batch_size=batch_size,
            n_steps=n_steps,
            omega=omega,
            omega_is=omega_is,
            n_continuous_vars=n_continuous_vars,
            n_discrete_vars=n_discrete_vars,
            n_discrete_vals=n_discrete_vals,
            learning_rate=learning_rate,
            adam_eps=adam_eps,
            beta_schedule=beta_schedule,
            tau_schedule=tau_schedule,
        )

        # @var encoder
        # Neural network that encodes observations into a distribution over
        # latent states.
        self.encoder = self.get_encoder_network(self.latent_type)

        # @var decoder
        # Neural network that decodes latent states into reconstructed
        # observations.
        self.decoder = self.get_decoder_network(self.latent_type)

        # @var optimizer
        # Adam optimizer for training both the encoder and decoder networks.
        self.optimizer = get_optimizer(
            [self.encoder, self.decoder], self.learning_rate, self.adam_eps
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
        loss, log_likelihood, kl_div = self.model_loss(obs, actions, next_obs)

        # Report the loss obtained for each sampled transition for potential
        # prioritization.
        loss = self.buffer.report(loss)
        loss = loss.mean()

        # Perform one step of gradient descent on the encoder and decoder
        # networks with gradient clipping.
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
            "kl_divergence": kl_div.mean(),
        }
        # @endcond

    def continuous_vfe(
        self, obs: Tensor, actions: Tensor, next_obs: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """!
        Compute the variational free energy for a continuous latent space.
        @param obs: the observations at time t
        @param actions: the actions at time t (unused)
        @param next_obs: the observations at time t + 1 (unused)
        @return a tuple containing the variational free energy, log-likelihood and KL-divergence
        """
        # @cond IGNORED_BY_DOXYGEN

        # Compute required tensors.
        mean_hat, log_var_hat = self.encoder(obs)
        state = self.reparam((mean_hat, log_var_hat))
        reconstructed_obs = self.decoder(state)

        # Compute the variational free energy.
        kl_div_hs = gauss_kl(mean_hat, log_var_hat)
        log_likelihood = self.likelihood_loss(obs, reconstructed_obs)
        vfe_loss = self.beta(self.current_step) * kl_div_hs - log_likelihood
        return vfe_loss, log_likelihood, kl_div_hs
        # @endcond

    def discrete_vfe(
        self, obs: Tensor, actions: Tensor, next_obs: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """!
        Compute the variational free energy for a discrete latent space.
        @param obs: the observations at time t
        @param actions: the actions at time t (unused)
        @param next_obs: the observations at time t + 1 (unused)
        @return a tuple containing the variational free energy, log-likelihood and KL-divergence
        """
        # @cond IGNORED_BY_DOXYGEN

        # Compute required tensors.
        tau = self.tau(self.current_step)
        logit_hats = self.encoder(obs)
        states = self.reparam(logit_hats, tau)
        reconstructed_obs = self.decoder(states)

        # Compute the variational free energy.
        kl_div = sum_cat_kl(logit_hats)
        log_likelihood = self.likelihood_loss(obs, reconstructed_obs)
        vfe_loss = self.beta(self.current_step) * kl_div - log_likelihood
        return vfe_loss, log_likelihood, kl_div
        # @endcond

    def mixed_vfe(
        self, obs: Tensor, actions: Tensor, next_obs: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """!
        Compute the variational free energy for a mixed latent space.
        @param obs: the observations at time t
        @param actions: the actions at time t (unused)
        @param next_obs: the observations at time t + 1 (unused)
        @return a tuple containing the variational free energy, log-likelihood and KL-divergence
        """
        # @cond IGNORED_BY_DOXYGEN

        # Compute required tensors.
        tau = self.tau(self.current_step)
        (mean_hat, log_var_hat), logit_hats = self.encoder(obs)
        states = self.reparam((mean_hat, log_var_hat), logit_hats, tau)
        reconstructed_obs = self.decoder(states)

        # Compute the variational free energy.
        kl_div_hs = gauss_kl(mean_hat, log_var_hat) + sum_cat_kl(logit_hats)
        log_likelihood = self.likelihood_loss(obs, reconstructed_obs)
        vfe_loss = self.beta(self.current_step) * kl_div_hs - log_likelihood
        return vfe_loss, log_likelihood, kl_div_hs
        # @endcond

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
        # @cond IGNORED_BY_DOXYGEN

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
                parameters = self.encoder(obs)
                states = self.reparam(parameters, tau=tau)
                reconstructed_obs = self.reconstructed_image_from(self.decoder(states))

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

            # Update the world model using the checkpoint.
            self.encoder = self.get_encoder_network(self.latent_type)
            safe_load_state_dict(self.encoder, checkpoint, "encoder")
            self.decoder = self.get_decoder_network(self.latent_type)
            safe_load_state_dict(self.decoder, checkpoint, "decoder")

            # Update the optimizer.
            self.optimizer = get_optimizer(
                [self.encoder, self.decoder],
                self.learning_rate,
                self.adam_eps,
                checkpoint,
            )
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
        return {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        } | super().as_dict()

    def save(
        self, checkpoint_name: str, buffer_checkpoint_name: str = ""
    ) -> None:
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

        # Save the replay buffer.
        self.buffer.save(checkpoint_path, buffer_checkpoint_name)
        # @endcond
