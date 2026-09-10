import logging
from datetime import datetime
from typing import List, Optional, SupportsFloat, Tuple

import relab
import torch
from gymnasium import Env
from relab.agents.AgentInterface import AgentInterface
from relab.agents.networks.CriticNetwork import ConvCriticNetwork
from relab.agents.networks.PolicyNetwork import ConvPolicyNetwork
from relab.helpers.Serialization import get_adam_optimizer, safe_load_state_dict
from relab.helpers.Typing import (
    ActionType,
    AttributeNames,
    Checkpoint,
    Config,
    ConfigInfo,
    ObservationType,
)
from torch import Tensor, nn
from torch.distributions.categorical import Categorical
from torch.nn import MSELoss


class PPO(AgentInterface):
    """!
    @brief Implements a Proximal Policy Optimization agent.

    @details
    This implementation is based on the paper:

    <b>Proximal Policy Optimization Algorithms</b>,
    published in arXiv, 2017.

    Authors:
    - John Schulman
    - Filip Wolski
    - Prafulla Dhariwal
    - Alec Radford
    - Oleg Klimov

    The paper introduced a new family of policy gradient algorithm called PPO.
    """

    def __init__(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        learning_rate: float = 0.00001,
        n_episodes: int = 10,
        n_epochs: int = 4,
        batch_size: int = 32,
        epsilon: float = 0.2,
        entropy_coefficient: float = 0.01,
        adam_eps: float = 1.5e-4,
        n_actions: int = 18,
        training: bool = True,
    ) -> None:
        """!
        Create a PPO agent.
        @param gamma: the discount factor
        @param gae_lambda: the trace decay of the generalized advantage estimate (1 for Monte-Carlo advantages)
        @param learning_rate: the learning rate
        @param n_episodes: the number of episode to sample per learning update
        @param n_epochs: the number of passes over the sampled episodes performed per learning update
        @param batch_size: the size of the mini-batches sampled from the collected episodes
        @param epsilon: the epsilon parameter of the PPO clip objective
        @param entropy_coefficient: the weight of the entropy bonus in the policy loss
        @param adam_eps: the epsilon parameter of the Adam optimizer
        @param n_actions: the number of actions available to the agent
        @param training: True if the agent is being trained, False otherwise
        """

        # Call the parent constructor.
        super().__init__(n_actions=n_actions, training=training)

        # @var gamma
        # Discount factor for future rewards (between 0 and 1).
        self.gamma = gamma

        # @var gae_lambda
        # Trace decay of the generalized advantage estimate (between 0 and 1).
        self.gae_lambda = gae_lambda

        # @var learning_rate
        # Learning rate for the optimizer.
        self.learning_rate = learning_rate

        # @var n_episodes
        # Number of episode to sample per learning update.
        self.n_episodes = n_episodes

        # @var n_epochs
        # Number of passes over the sampled episodes performed per learning update.
        self.n_epochs = n_epochs

        # @var batch_size
        # Number of experiences in the mini-batches sampled from the collected episodes.
        self.batch_size = batch_size

        # @var epsilon
        # The epsilon parameter of the PPO clip objective.
        self.epsilon = epsilon

        # @var entropy_coefficient
        # Weight of the entropy bonus added to the policy loss.
        self.entropy_coefficient = entropy_coefficient

        # @var adam_eps
        # Epsilon parameter for the Adam optimizer.
        self.adam_eps = adam_eps

        # @var policy_net
        # The policy network that predicts the distribution over actions.
        self.policy_net = self.get_policy_network()

        # @var value_net
        # The value network that predicts the value of each state.
        self.value_net = self.get_value_network()

        # @var policy_optimizer
        # Adam optimizer for training the policy network.
        self.policy_optimizer = get_adam_optimizer(
            [self.policy_net],
            self.learning_rate,
            self.adam_eps,
        )

        # @var value_optimizer
        # Adam optimizer for training the value network.
        self.value_optimizer = get_adam_optimizer(
            [self.value_net],
            self.learning_rate,
            self.adam_eps,
        )

    def get_policy_network(self) -> nn.Module:
        """
        Retrieve the policy network of the PPO agent.
        :return: the policy network.
        """
        network = ConvPolicyNetwork(n_actions=self.n_actions)
        network.train(self.training)
        network.to(self.device)
        return network

    def get_value_network(self) -> nn.Module:
        """
        Retrieve the value network of the PPO agent.
        :return: the value network.
        """
        network = ConvCriticNetwork(n_outputs=1)
        network.train(self.training)
        network.to(self.device)
        return network

    def step_with_log_prob(self, obs: ObservationType) -> Tuple[ActionType, Tensor]:
        """!
        Select the next action to perform in the environment.
        @param obs: the observation available to make the decision
        @return the next action to perform and the log-probability of the next action
        """
        logits = self.policy_net(obs)
        distribution = Categorical(logit=logits)
        action = distribution.sample()
        return action.item(), distribution.log_prob(action)

    def step(self, obs: ObservationType) -> ActionType:
        """!
        Select the next action to perform in the environment.
        @param obs: the observation available to make the decision
        @return the next action to perform
        """
        return self.step_with_log_prob(obs)[0]

    def compute_advantages(
        self,
        rewards: List[SupportsFloat],
        values: List[float],
        last_value: float = 0.0,
    ) -> Tuple[List[float], List[float]]:
        """
        Compute the generalized advantage estimates of an episode, and the returns they imply.
        Note that for a trace decay of one, the returns are the discounted sum of future rewards.
        @param rewards: the rewards retrieved during the episode
        @param values: the values predicted by the value network during the episode
        @param last_value: the value of the observation following the episode (zero if the episode terminated)
        @return a tuple (advantages, returns)
        """

        # Accumulate the temporal difference errors backward in time.
        advantages = []
        advantage = 0.0
        next_value = last_value
        for reward, value in zip(reversed(rewards), reversed(values)):
            delta = float(reward) + self.gamma * next_value - value
            advantage = delta + self.gamma * self.gae_lambda * advantage
            advantages.insert(0, advantage)
            next_value = value

        # The returns are the targets of the value network, i.e., the advantages
        # corrected by the (biased) values predicted by the value network.
        returns = [advantage + value for advantage, value in zip(advantages, values)]
        return advantages, returns

    def rollouts(
        self, env: Env, config: ConfigInfo
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Run rollouts to collect a batch of data.
        Note that the observations are kept on the CPU, as an entire batch of episodes
        may not fit on the GPU, the mini-batches are moved to the GPU in the learn function.
        @param env: the environment in which the episodes are run
        @param config: the training configuration
        @return a tuple (observations, actions, log-probability of actions, advantages, returns)
        """

        # Collect the requested number of episodes.
        observations = []
        actions = []
        log_probs = []
        advantages = []
        returns = []
        for i in range(self.n_episodes):

            # Retrieve the initial observation from the environment.
            obs, _ = env.reset()

            # Collect a single episode.
            rewards = []
            values = []
            terminated = False
            done = False
            while not done:

                # Collect the observations.
                observations.append(torch.unsqueeze(obs, dim=0))

                # Perform one step in the environment.
                # The gradients are not required, the policy and value networks
                # are re-evaluated on the collected batch in the learn function.
                with torch.no_grad():
                    device_obs = obs.to(self.device)
                    action, log_prob = self.step_with_log_prob(device_obs)
                    value = self.value_net(device_obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # Collect the actions, log-probabilities, values and rewards.
                actions.append(action)
                log_probs.append(log_prob)
                values.append(value.item())
                rewards.append(reward)

                # Save the agent (if needed).
                if self.current_step % config["checkpoint_frequency"] == 0:
                    self.save(f"model_{self.current_step}.pt")

                # Log the mean episodic reward in tensorboard (if needed).
                self.report(reward, done)
                if self.current_step % config["tensorboard_log_interval"] == 0:
                    self.log_performance_in_tensorboard()

                # Increase the number of training steps done.
                self.current_step += 1

            # If the episode was truncated instead of terminated, the rewards
            # following the last observation are estimated by the value network.
            last_value = 0.0
            if not terminated:
                with torch.no_grad():
                    last_value = self.value_net(obs.to(self.device)).item()

            # Compute the advantages and returns of the episode.
            episode_advantages, episode_returns = self.compute_advantages(
                rewards, values, last_value
            )
            advantages += episode_advantages
            returns += episode_returns

        # Format output tensors and return them.
        observations = torch.cat(observations)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        log_probs = torch.cat(log_probs).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        return observations, actions, log_probs, advantages, returns

    def compute_value_loss(self, obs: Tensor, returns: Tensor) -> Tensor:
        """
        Compute the loss function of the value network.
        @param obs: the observations whose values must be predicted
        @param returns: the returns, i.e., the targets of the value network
        @return the loss function
        """
        loss_fc = MSELoss()
        return loss_fc(torch.squeeze(self.value_net(obs), dim=1), returns)

    def compute_policy_loss(
        self, obs: Tensor, actions: Tensor, log_probs: Tensor, advantages: Tensor
    ) -> Tensor:
        """
        Compute the clipped surrogate loss function of the policy network.
        @param obs: the observations available to make the decisions
        @param actions: the actions that were performed
        @param log_probs: the (old) log-probabilities of the actions
        @param advantages: the advantages of the observation-action pairs
        @return the loss function
        """

        # Compute the ratio between the probabilities of the actions according
        # to the current policy and the policy that collected the episodes.
        logits = self.policy_net(obs)
        distribution = Categorical(logits=logits)
        ratio = torch.exp(distribution.log_prob0(actions) - log_probs)

        # Compute the clipped objective, which prevents the current policy from
        # moving too far away from the policy that collected the episodes.
        clipped_ratio = torch.clip(ratio, 1 - self.epsilon, 1 + self.epsilon)
        loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # Add an entropy bonus encouraging the policy to keep exploring.
        return loss - self.entropy_coefficient * distribution.entropy().mean()

    def learn(
        self,
        observations: Tensor,
        actions: Tensor,
        log_probs: Tensor,
        advantages: Tensor,
        returns: Tensor,
    ) -> None:
        """!
        Perform several epochs of gradient descent on the policy and value networks.
        @param observations: the observations of the collected episodes (on the CPU)
        @param actions: the actions performed during the collected episodes
        @param log_probs: the log-probabilities of the actions when they were performed
        @param advantages: the advantages of the observation-action pairs
        @param returns: the returns, i.e., the targets of the value network
        """

        # Normalize the advantages to reduce the variance of the policy gradient.
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / advantages.std().clamp(1e-8)

        # Perform several passes over the collected episodes.
        n_experiences = observations.shape[0]
        for epoch in range(self.n_epochs):

            # Shuffle the experiences to decorrelate the mini-batches.
            indices = torch.randperm(n_experiences)
            for start in range(0, n_experiences, self.batch_size):

                # Retrieve the next mini-batch.
                end = start + self.batch_size
                batch = indices[start:end]
                obs = observations[batch].to(self.device)

                # Perform one step of gradient descent on the value network with
                # gradient clipping.
                loss = self.compute_value_loss(obs, returns[batch])
                self.value_optimizer.zero_grad()
                loss.backward()
                for param in self.value_net.parameters():
                    param.grad.data.clamp_(-1, 1)
                self.value_optimizer.step()

                # Perform one step of gradient descent on the policy network with
                # gradient clipping.
                loss = self.compute_policy_loss(
                    obs, actions[batch], log_probs[batch], advantages[batch]
                )
                self.policy_optimizer.zero_grad()
                loss.backward()
                for param in self.policy_net.parameters():
                    param.grad.data.clamp_(-1, 1)
                self.policy_optimizer.step()

    def train(self, env: Env) -> None:
        """!
        Train the agent in the gym environment passed as parameters
        @param env: the gym environment
        """
        # @cond IGNORED_BY_DOXYGEN

        # Train the agent.
        config = relab.config()
        logging.info(f"Start the training at {datetime.now()}")
        while self.current_step < config["max_n_steps"]:

            # Collect the trajectories on which the networks are trained.
            observations, actions, log_probs, advantages, returns = self.rollouts(
                env, config
            )

            # Improve the policy and value networks using the collected trajectories.
            self.learn(observations, actions, log_probs, advantages, returns)

        # Save the final version of the model.
        self.save(f"model_{config['max_n_steps']}.pt")

        # Close the environment.
        env.close()
        # @endcond

    def load(
        self,
        checkpoint_name: str = "",
        buffer_checkpoint_name: str = "",
        attr_names: Optional[AttributeNames] = None,
    ) -> Checkpoint:
        """!
        Load an agent from the filesystem.
        @param checkpoint_name: the name of the agent checkpoint to load
        @param buffer_checkpoint_name: the name of the replay buffer checkpoint to load ("" for default name)
        @param attr_names: a list of attribute names to load from the checkpoint (load all attributes by default)
        @return the loaded checkpoint object
        """
        # @cond IGNORED_BY_DOXYGEN
        try:
            # Call the parent load function.
            checkpoint = super().load(
                checkpoint_name, buffer_checkpoint_name, self.as_dict().keys()
            )

            # Update the agent's networks using the checkpoint.
            self.policy_net = self.get_policy_network()
            safe_load_state_dict(self.policy_net, checkpoint, "policy_net")
            self.value_net = self.get_value_network()
            safe_load_state_dict(self.value_net, checkpoint, "value_net")

            # Update the optimizers.
            self.policy_optimizer = get_adam_optimizer(
                [self.policy_net],
                self.learning_rate,
                self.adam_eps,
                checkpoint,
                "policy_optimizer",
            )
            self.value_optimizer = get_adam_optimizer(
                [self.value_net],
                self.learning_rate,
                self.adam_eps,
                checkpoint,
                "value_optimizer",
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
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "learning_rate": self.learning_rate,
            "n_episodes": self.n_episodes,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "epsilon": self.epsilon,
            "entropy_coefficient": self.entropy_coefficient,
            "adam_eps": self.adam_eps,
            "policy_net": self.policy_net.state_dict(),
            "value_net": self.value_net.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
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
        @param buffer_checkpoint_name: the name of the checkpoint to save the replay buffer ("" for default name)
        @param agent_conf: a dictionary representing the agent's attributes to be saved (for internal use only)
        """
        super().save(checkpoint_name, buffer_checkpoint_name, self.as_dict())
