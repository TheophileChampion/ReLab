import random
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium import Env
from matplotlib.figure import Figure
from relab.agents.AgentInterface import ReplayType
from relab.agents.networks.CriticNetwork import LinearCriticNetwork
from relab.agents.schedule.ExponentialSchedule import ExponentialSchedule
from relab.agents.VariationalModel import (
    LikelihoodType,
    VariationalModel,
)
from relab.helpers.MatPlotLib import MatPlotLib
from relab.helpers.Serialization import get_adam_optimizer, safe_load_state_dict
from relab.helpers.Typing import (
    ActionType,
    AttributeNames,
    Checkpoint,
    Config,
    ObservationType,
)
from relab.helpers.VariationalInference import gaussian_kl_divergence as kl_gauss
from relab.helpers.VariationalInference import (
    gaussian_reparameterization,
)
from torch import Tensor, nn, unsqueeze


class CHMM(VariationalModel):
    """!
    @brief Implements a Critical Hidden Markov Model.

    @details
    This implementation is based on the paper:

    <b>Deconstructing Deep Active Inference: A Contrarian Information Gatherer</b>,
    published in Neural Computation, 2024.

    Authors:
    - Champion T.
    - Grześ M.
    - Bonheme L.
    - Bowman H.

    More precisely, the CHMM extends the HMM framework by introducing a critic
    network for action selection. Note, this agent takes actions based on the
    expected reward or expected free energy.
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
        epsilon_schedule: Any = None,
        gamma: float = 0.99,
        reward_only: bool = True,
        n_steps_between_synchro: int = 10000,
        reward_coefficient: int = 500,
    ) -> None:
        """!
        Create a Critical Hidden Markov Model agent taking actions based on the
        expected reward or expected free energy.
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
        @param epsilon_schedule: the exponential schedule used by the epsilon greedy algorithm
        @param gamma: the discount factor from reinforcement learning
        @param reward_only: whether the agent must maximize reward only or expected free energy
        @param n_steps_between_synchro: the number of steps between two synchronisations
            of the target and the critic
        @param reward_coefficient: the coefficient by which the reward is multiplied
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

        # @var epsilon_schedule
        # Schedule for the epsilon threshold used by the epsilon greedy algorithm.
        self.epsilon_schedule = (
            (1.0, 0.0, 1e5) if epsilon_schedule is None else epsilon_schedule
        )

        # @var epsilon_threshold
        # Scheduler for the epsilon threshold used by the epsilon greedy algorithm.
        self.epsilon_threshold = ExponentialSchedule(self.epsilon_schedule)

        # @var gamma
        # Discount factor from reinforcement learning.
        self.gamma = gamma

        # @var reward_only
        # Whether the agent must maximize reward only or expected free energy.
        self.reward_only = reward_only

        # @var n_steps_between_synchro
        # The number of steps between two synchronisations of the target and the critic.
        self.n_steps_between_synchro = n_steps_between_synchro

        # @var reward_coefficient
        # The coefficient by which the reward is multiplied in the critic loss.
        self.reward_coefficient = reward_coefficient

        # @var encoder
        # Neural network that encodes observations into latent states.
        self.encoder = self.get_encoder_network()

        # @var decoder
        # Neural network that decodes latent states into observations.
        self.decoder = self.get_decoder_network()

        # @var transition_net
        # Neural network that models the transition dynamics in latent space.
        self.transition_net = self.get_transition_network()

        # @var critic
        # Neural network that predict the expected reward or expected free energy.
        self.critic = self.get_critic_network()

        # @var target
        # Neural network that predict the expected reward or expected free energy.
        self.target = deepcopy(self.critic)
        self.target.eval()

        # @var optimizer
        # Adam optimizer for training the encoder, decoder, and transition networks.
        self.optimizer = get_adam_optimizer(
            [self.encoder, self.decoder, self.transition_net],
            self.learning_rate,
            self.adam_eps,
        )

        # @var optimizer_efe
        # Adam optimizer for training the critic network.
        self.optimizer_efe = get_adam_optimizer(
            [self.encoder, self.critic],
            self.learning_rate,
            self.adam_eps,
        )

    def get_critic_network(self):
        """!
        Retrieve the critic network.
        @return the critic network
        """
        # @cond IGNORED_BY_DOXYGEN
        critic = LinearCriticNetwork(
            n_actions=self.n_actions, n_continuous_vars=self.n_cont_vars
        )
        critic.train(self.training)
        critic.to(self.device)
        return critic
        # @endcond

    def step(self, obs: ObservationType) -> ActionType:
        """!
        Select the next action to perform in the environment.
        @param obs: the observation available to make the decision
        @return the next action to perform
        """

        # Extract the current state quality from the current observation.
        obs = torch.unsqueeze(obs, dim=0)
        state, _ = self.encoder(obs)
        quality = self.critic(state)

        # Sample a number between 0 and 1, and either execute a random action or
        # the reward maximizing action according to the sampled value.
        if random.random() > self.epsilon_threshold(self.current_step):
            return quality.max(1)[1].item()
        return np.random.choice(quality.shape[1])

    def learn(self) -> Optional[Dict[str, Any]]:
        """!
        Perform one step of gradient descent on the world model.
        @return the loss of the sampled batch
        """
        # @cond IGNORED_BY_DOXYGEN

        # Synchronize the target with the critic (if needed).
        if self.current_step % self.n_steps_between_synchro == 0:
            self.synchronize_target()

        # Sample the replay buffer.
        obs, actions, rewards, dones, next_obs = self.buffer.sample()

        # Compute the variational free energy.
        vfe, log_likelihood, kl_div = self.vfe(obs, actions, next_obs)

        # Report the loss obtained for each sampled transition for potential
        # prioritization.
        vfe = self.buffer.report(vfe)

        # Perform one step of gradient descent on the encoder, decoder, and transition
        # networks with gradient clipping.
        self.optimizer.zero_grad()
        vfe = vfe.mean()
        vfe.backward()
        for param in self.encoder.parameters():
            param.grad.data.clamp_(-1, 1)
        for param in self.decoder.parameters():
            param.grad.data.clamp_(-1, 1)
        for param in self.transition_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Compute the expected free energy loss.
        efe_loss = self.efe_loss(obs, actions, rewards, dones, next_obs)

        # Perform one step of gradient descent on the encoder and critic
        # networks with gradient clipping on the critic.
        self.optimizer_efe.zero_grad()
        efe_loss = efe_loss.mean()
        efe_loss.backward()
        for param in self.critic.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer_efe.step()

        return {
            "vfe": vfe.mean(),
            "beta": self.beta(self.current_step),
            "log_likelihood": log_likelihood.mean(),
            "kl_divergence": kl_div.mean(),
            "efe_loss": efe_loss.mean(),
        }
        # @endcond

    def synchronize_target(self):
        """
        Synchronize the target with the critic.
        """
        self.target = deepcopy(self.critic)
        self.target.eval()

    def vfe(
        self, obs: Tensor, actions: Tensor, next_obs: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """!
        Compute the variational free energy of the Critical HMM.
        @param obs: the observations at time t
        @param actions: the actions at time t
        @param next_obs: the observations at time t + 1
        @return a triple containing the (variational free energy, log-likelihood, kl-divergence)
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

    def efe_loss(
        self,
        obs: Tensor,
        actions: Tensor,
        rewards: Tensor,
        dones: Tensor,
        next_obs: Tensor,
    ) -> Tensor:
        """!
        Compute the critic's loss function.
        @param obs: the observations at time t
        @param actions: the actions at time t
        @param rewards: the rewards received
        @param dones: whether the environment stop after performing the actions
        @param next_obs: the observations at time t + 1
        @return the loss function
        """

        # Compute required vectors.
        mean_hat, log_var_hat = self.encoder(obs)
        states = gaussian_reparameterization(mean_hat, log_var_hat)
        next_mean_hat, next_log_var_hat = self.encoder(next_obs)
        mean, log_var = self.transition_net(states, actions)

        # Compute the G-values of each action in the current state.
        critic_pred = self.critic(mean_hat)
        critic_pred = critic_pred.gather(
            dim=1, index=unsqueeze(actions.to(torch.int64), dim=1)
        )

        # For each batch entry where the simulation did not stop,
        # compute the value of the next states.
        future_gval = torch.zeros(self.batch_size, device=self.device)
        future_gval[torch.logical_not(dones)] = self.target(
            next_mean_hat[torch.logical_not(dones)]
        ).max(1)[0]

        # Compute the discounted G values.
        immediate_gval = self.reward_coefficient * rewards
        if self.reward_only is False:
            immediate_gval += kl_gauss(next_mean_hat, next_log_var_hat, mean, log_var)
        gval = immediate_gval.to(torch.float32) + self.gamma * future_gval

        # Compute the expected free energy loss.
        loss = nn.SmoothL1Loss()
        efe_loss = loss(critic_pred, gval.detach().unsqueeze(dim=1))
        return efe_loss

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
            obs = obs.to(self.device)
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
                obs = obs.to(self.device)
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

            # Ensure the epsilon threshold follows the loaded schedule.
            self.epsilon_threshold = ExponentialSchedule(self.epsilon_schedule)

            # Update the world model using the checkpoint.
            self.encoder = self.get_encoder_network()
            safe_load_state_dict(self.encoder, checkpoint, "encoder")
            self.decoder = self.get_decoder_network()
            safe_load_state_dict(self.decoder, checkpoint, "decoder")
            self.transition_net = self.get_transition_network()
            safe_load_state_dict(self.transition_net, checkpoint, "transition_net")
            self.critic = self.get_critic_network()
            safe_load_state_dict(self.critic, checkpoint, "critic")
            self.target = deepcopy(self.critic)
            safe_load_state_dict(self.target, checkpoint, "target")

            # Update the optimizers.
            self.optimizer = get_adam_optimizer(
                [self.encoder, self.decoder, self.transition_net],
                self.learning_rate,
                self.adam_eps,
                checkpoint,
            )
            self.optimizer_efe = get_adam_optimizer(
                [self.encoder, self.critic],
                self.learning_rate,
                self.adam_eps,
                checkpoint,
                "optimizer_efe",
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
            "critic": self.critic.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "optimizer_efe": self.optimizer_efe.state_dict(),
            "gamma": self.gamma,
            "reward_only": self.reward_only,
            "epsilon_schedule": self.epsilon_schedule,
            "n_steps_between_synchro": self.n_steps_between_synchro,
            "reward_coefficient": self.reward_coefficient,
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
