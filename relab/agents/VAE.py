import logging
import os
import re
from datetime import datetime
from os.path import join
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium import Env
from matplotlib.figure import Figure
from relab import relab
from relab.agents.AgentInterface import ReplayType
from relab.agents.VariationalModel import LikelihoodType, VariationalModel
from relab.cpp.agents.memory import Experience
from relab.helpers.MatPlotLib import MatPlotLib
from relab.helpers.Serialization import get_adam_optimizer, safe_load_state_dict
from relab.helpers.Typing import (
    ActionType,
    AttributeNames,
    Checkpoint,
    Config,
    ObservationType,
)
from relab.helpers.VariationalInference import gaussian_kl_divergence as gauss_kl
from relab.helpers.VariationalInference import (
    gaussian_reparameterization,
)
from torch import Tensor


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
        learning_starts: int = 50000,
        n_actions: int = 18,
        training: bool = True,
        likelihood_type: LikelihoodType = LikelihoodType.BERNOULLI,
        n_continuous_vars: int = 15,
        learning_rate: float = 0.0001,
        adam_eps: float = 1e-8,
        beta_schedule: Any = None,
        replay_type: ReplayType = ReplayType.PRIORITIZED,
        buffer_size: int = 1000000,
        batch_size: int = 50,
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
        @param n_continuous_vars: the number of continuous latent variables
        @param learning_rate: the learning rate
        @param adam_eps: the epsilon parameter of the Adam optimizer
        @param beta_schedule: the piecewise linear schedule of the KL-divergence weight of beta-VAE
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
            buffer_size=buffer_size,
            batch_size=batch_size,
            n_steps=n_steps,
            omega=omega,
            omega_is=omega_is,
            n_continuous_vars=n_continuous_vars,
            learning_rate=learning_rate,
            adam_eps=adam_eps,
            beta_schedule=beta_schedule,
        )

        # @var encoder
        # Neural network that encodes observations into a distribution over
        # latent states.
        self.encoder = self.get_encoder_network()

        # @var decoder
        # Neural network that decodes latent states into reconstructed
        # observations.
        self.decoder = self.get_decoder_network()

        # @var optimizer
        # Adam optimizer for training both the encoder and decoder networks.
        self.optimizer = get_adam_optimizer(
            [self.encoder, self.decoder], self.learning_rate, self.adam_eps
        )

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

    def learn(self) -> Optional[Dict[str, Any]]:
        """!
        Perform one step of gradient descent on the world model.
        @return the loss of the sampled batch
        """
        # @cond IGNORED_BY_DOXYGEN

        # Sample the replay buffer.
        obs, actions, _, _, next_obs = self.buffer.sample()

        # Compute the model loss.
        loss, log_likelihood, kl_div = self.vfe(obs, actions, next_obs)

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

    def vfe(
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
        state = gaussian_reparameterization(mean_hat, log_var_hat)
        reconstructed_obs = self.decoder(state)

        # Compute the variational free energy.
        kl_div_hs = gauss_kl(mean_hat, log_var_hat)
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

        # Create the figure and the grid specification.
        height, width = grid_size
        n_cols = 2
        fig = plt.figure(figsize=(width + n_cols, height * 2))
        gs = fig.add_gridspec(height * 2, width + n_cols)

        # Iterate over the grid's rows.
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
                mean_hat, log_var_hat = self.encoder(obs)
                states = gaussian_reparameterization(mean_hat, log_var_hat)
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
        # @cond IGNORED_BY_DOXYGEN
        try:
            # Call the parent load function.
            checkpoint = super().load(
                checkpoint_name, buffer_checkpoint_name, self.as_dict().keys()
            )

            # Update the world model using the checkpoint.
            self.encoder = self.get_encoder_network()
            safe_load_state_dict(self.encoder, checkpoint, "encoder")
            self.decoder = self.get_decoder_network()
            safe_load_state_dict(self.decoder, checkpoint, "decoder")

            # Update the optimizer.
            self.optimizer = get_adam_optimizer(
                [self.encoder, self.decoder],
                self.learning_rate,
                self.adam_eps,
                checkpoint,
            )
            return checkpoint

        # Catch the exception raise if the checkpoint could not be located.
        except FileNotFoundError:
            return None
        # @endcond

    def as_dict(self) -> Config:
        """!
        Convert the agent into a dictionary that can be saved on the filesystem.
        @return the dictionary
        """
        return {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
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
        super().save(checkpoint_name, buffer_checkpoint_name, self.as_dict())
