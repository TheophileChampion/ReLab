from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from gymnasium import Env
from matplotlib.figure import Figure
from relab.agents.AgentInterface import ReplayType
from relab.agents.VariationalModel import (
    LikelihoodType,
    VariationalModel,
)
from relab.helpers.MatPlotLib import MatPlotLib
from relab.helpers.Serialization import get_optimizer, safe_load_state_dict
from relab.helpers.Typing import AttributeNames, Checkpoint, Config
from relab.helpers.VariationalInference import gaussian_kl_divergence as kl_gauss
from relab.helpers.VariationalInference import (
    gaussian_reparameterization,
)
from torch import Tensor


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
        Create a Hidden Markov Model agent taking random actions.
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
        # Neural network that encodes observations into latent states.
        self.encoder = self.get_encoder_network()

        # @var decoder
        # Neural network that decodes latent states into observations.
        self.decoder = self.get_decoder_network()

        # @var transition
        # Neural network that models the transition dynamics in latent space.
        self.transition_net = self.get_transition_network()

        # @var optimizer
        # Adam optimizer for training the encoder, decoder, and transition networks.
        self.optimizer = get_optimizer(
            [self.encoder, self.decoder, self.transition_net],
            self.learning_rate,
            self.adam_eps,
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
        @param actions: the actions at time t
        @param next_obs: the observations at time t + 1
        @return the variational free energy
        """

        # Compute required vectors.
        mean_hat, log_var_hat = self.encoder(obs)
        states = gaussian_reparameterization(mean_hat, log_var_hat)
        reconstructed_obs = self.decoder(states)
        next_mean_hat, next_log_var_hat = self.encoder(next_obs)
        next_states = gaussian_reparameterization(next_mean_hat, next_log_var_hat)
        next_reconstructed_obs = self.decoder(next_states)
        mean, log_var = self.transition_net(states, actions)

        # Compute the variational free energy.
        kl_div_hs = kl_gauss(mean_hat, log_var_hat)
        kl_div_hs += kl_gauss(next_mean_hat, next_log_var_hat, mean, log_var)
        log_likelihood = self.likelihood_loss(obs, reconstructed_obs)
        log_likelihood += self.likelihood_loss(next_obs, next_reconstructed_obs)
        vfe_loss = self.beta(self.current_step) * kl_div_hs - log_likelihood
        return vfe_loss, log_likelihood, kl_div_hs

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
        h = 0
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
            obs = torch.unsqueeze(obs, dim=0).to(self.device)
            mean, log_var = self.encoder(obs)
            states = gaussian_reparameterization(mean, log_var)
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

                # Execute the agent's action in the environment to obtain the
                # next ground truth observation.
                action = self.step(obs)
                obs, _, terminated, truncated, _ = env.step(action)
                obs = torch.unsqueeze(obs, dim=0)
                done = terminated or truncated
                action = torch.tensor([action]).to(self.device)

                # Simulate the agent's action to obtain the next reconstructed
                # observation.
                mean, log_var = self.transition_net(states, action)
                states = gaussian_reparameterization(mean, log_var)
                reconstructed_obs = self.reconstructed_image_from(self.decoder(states))

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
            self.transition_net = self.get_transition_network()
            safe_load_state_dict(self.transition_net, checkpoint, "transition_net")

            # Update the optimizer.
            self.optimizer = get_optimizer(
                [self.encoder, self.decoder, self.transition_net],
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
            "transition_net": self.transition_net.state_dict(),
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
