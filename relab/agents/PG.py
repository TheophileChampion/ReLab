import logging
from datetime import datetime
from enum import IntEnum
from typing import Optional, Tuple, List, SupportsFloat

import relab
import torch
from torch import Tensor
from torch import nn
from torch.distributions.categorical import Categorical
from gymnasium import Env
from relab.agents.AgentInterface import AgentInterface

from relab.agents.networks.PolicyNetwork import ConvPolicyNetwork
from relab.helpers.Serialization import get_adam_optimizer, safe_load_state_dict
from relab.helpers.Typing import (
    ActionType,
    AttributeNames,
    Checkpoint,
    Config,
    ObservationType, ConfigInfo,
)


class PolicyGradientType(IntEnum):
    """!
    The type of policy gradient algorithm to use.
    """

    # @var VANILLA
    # The simplest form of policy gradient algorithm.
    VANILLA = 0

    # @var NORMALIZED
    # Future returns are normalized by subtracting the mean and dividing by the standard deviation.
    NORMALIZED = 1

    # @var ROBUST
    # Normalized future returns plus trick to make the algorithm robust to the episode length.
    ROBUST = 2


class PG(AgentInterface):
    """!
    @brief Implements a policy gradient agent.

    @details
    This implementation is based on the paper:

    <b>Policy Gradient Methods for Reinforcement Learning with Function Approximation</b>,
    published in NeurIPS, 1999.

    Authors:
    - Richard S. Sutton
    - David A. McAllester
    - Satinder P. Singh
    - Yishay Mansour

    The paper discusses the family of policy gradient algorithms.
    For another clear exposition of policy gradient, see:
    https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html
    """

    def __init__(
        self,
        gamma: float = 0.99,
        learning_rate: float = 0.00001,
        n_episodes: int = 10,
        adam_eps: float = 1.5e-4,
        n_actions: int = 18,
        pg_type: PolicyGradientType = PolicyGradientType.ROBUST,
        training: bool = True,
    ) -> None:
        """!
        Create a policy gradient agent.
        @param gamma: the discount factor
        @param learning_rate: the learning rate
        @param n_episodes: the number of episodes run per learning update
        @param adam_eps: the epsilon parameter of the Adam optimizer
        @param n_actions: the number of actions available to the agent
        @param pg_type: the type of policy gradient algorithm to use
        @param training: True if the agent is being trained, False otherwise
        """

        # Call the parent constructor.
        super().__init__(n_actions=n_actions, training=training)

        # @var gamma
        # Discount factor for future rewards (between 0 and 1).
        self.gamma = gamma

        # @var learning_rate
        # Learning rate for the optimizer.
        self.learning_rate = learning_rate

        # @var n_episodes
        # Number of episodes run per learning update.
        self.n_episodes = n_episodes

        # @var adam_eps
        # Epsilon parameter for the Adam optimizer.
        self.adam_eps = adam_eps

        # @var pg_type
        # The type of policy gradient algorithm to use.
        self.pg_type = pg_type

        # @var policy_net
        # The policy network that predicts the distribution over actions.
        self.policy_net = self.get_policy_network()

        # @var optimizer
        # Adam optimizer for training the policy network.
        self.optimizer = get_adam_optimizer(
            [self.policy_net],
            self.learning_rate,
            self.adam_eps,
        )

    def get_policy_network(self) -> nn.Module:
        """
        Retrieve the policy network of the policy gradient agent.
        @return the policy network
        """
        network = ConvPolicyNetwork()
        network.train(self.training)
        network.to(self.device)
        return network

    def compute_future_returns(self, rewards: List[SupportsFloat]) -> List[SupportsFloat]:
        """
        Compute the future discounted returns.
        @param rewards: the rewards retrieved during an episode
        @return the future returns
        """
        future_returns = []
        future_return = 0
        for reward in reversed(rewards):
            future_return = float(reward) + future_return * self.gamma
            future_returns.insert(0, future_return)
        return future_returns

    def step(self, obs: ObservationType) -> Tuple[ActionType, Tensor]:
        """!
        Select the next action to perform in the environment.
        @param obs: the observation available to make the decision
        @return a tuple containing the next action to perform and its log-probability
        """
        probs = self.policy_net(obs)
        distribution = Categorical(probs)
        action = distribution.sample()
        return action, distribution.log_prob(action)

    def rollouts(self, env: Env, config: ConfigInfo) -> Tuple[Tensor, Tensor]:
        """
        Run rollouts to collect a batch of data.
        @param env: the environment in which the episodes are run
        @param config: the training configuration
        @return a tuple (log-probability of actions, future_returns)
        """

        # Collect the requested number of episodes.
        future_returns = []
        log_probs = []
        for i in range(self.n_episodes):

            # Retrieve the initial observation from the environment.
            obs, _ = env.reset()
            obs = obs.to(self.device)

            # Collect a single episode.
            rewards = []
            done = False
            while not done:

                # Perform one step in the environment.
                action, log_prob = self.step(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                obs = obs.to(self.device)
                done = terminated or truncated

                # Collect the log-probabilities and rewards.
                log_probs.append(log_prob)
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

            # Compute the future returns.
            future_returns += self.compute_future_returns(rewards)

        # Format output tensors and return them.
        log_probs = torch.cat(log_probs).to(self.device)
        future_returns = torch.tensor(future_returns).to(self.device)
        return log_probs, future_returns

    def compute_loss(self, log_probs: Tensor, future_returns: Tensor) -> Tensor:
        """
        Compute the policy gradient loss.
        :param log_probs: the log-probability of the selected actions
        :param future_returns: the future returns obtained by following the current policy
        :return: the loss
        """

        # Normalise the future returns, if required.
        if self.pg_type == PolicyGradientType.NORMALIZED or self.pg_type == PolicyGradientType.ROBUST:
            mean = future_returns.mean()
            std = future_returns.std()
            future_returns = (future_returns - mean) / std.clamp(1e-8)

        # Compute the loss.
        if self.pg_type == PolicyGradientType.ROBUST:
            loss = -(log_probs * future_returns).mean()
        else:
            loss = -(log_probs * future_returns).sum() / self.n_episodes
        return loss

    def train(self, env: Env) -> None:
        """!
        Train the agent in the gym environment passed as parameters.
        @param env: the gym environment
        """
        # @cond IGNORED_BY_DOXYGEN

        # Train the agent.
        config = relab.config()
        logging.info(f"Start the training at {datetime.now()}")
        while self.current_step < config["max_n_steps"]:

            # Collect the next batch on which the policy network will be trained.
            log_probs, future_returns = self.rollouts(env, config)

            # Perform one step of gradient descent to train the policy network.
            self.optimizer.zero_grad()
            loss = self.compute_loss(log_probs, future_returns)
            loss.backward()
            self.optimizer.step()

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

            # Update the optimizers.
            self.optimizer = get_adam_optimizer(
                [self.policy_net], self.learning_rate, self.adam_eps, checkpoint
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
            "learning_rate": self.learning_rate,
            "n_episodes": self.n_episodes,
            "adam_eps": self.adam_eps,
            "pg_type": self.pg_type,
            "policy_net": self.policy_net.state_dict(),
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
        @param buffer_checkpoint_name: the name of the checkpoint to save the replay buffer ("" for default name)
        @param agent_conf: a dictionary representing the agent's attributes to be saved (for internal use only)
        """
        super().save(checkpoint_name, buffer_checkpoint_name, self.as_dict())
