import logging
from datetime import datetime
from typing import Optional, Tuple, List, SupportsFloat

import numpy as np
from torch.nn import MSELoss

import relab
import torch
from gymnasium import Env
from relab.agents.AgentInterface import AgentInterface
from relab.cpp.agents.memory import Experience

from relab.agents.networks.CriticNetwork import ConvCriticNetwork
from relab.agents.networks.PolicyNetwork import ConvPolicyNetwork
from relab.helpers.Serialization import get_adam_optimizer, safe_load_state_dict
from relab.helpers.Typing import (
    ActionType,
    AttributeNames,
    Checkpoint,
    Config,
    ObservationType,
)


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
        learning_rate: float = 0.00001,
        n_episodes: int = 10,
        epsilon: float = 0.2,
        adam_eps: float = 1.5e-4,
        n_actions: int = 18,
        training: bool = True,
    ) -> None:
        """!
        Create a PPO agent.
        @param gamma: the discount factor
        @param learning_rate: the learning rate
        @param n_episodes: the number of episode to sample per learning update
        @param epsilon: the epsilon parameter of the PPO clip objective
        @param adam_eps: the epsilon parameter of the Adam optimizer
        @param n_actions: the number of actions available to the agent
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
        # Number of episode to sample per learning update.
        self.n_episodes = n_episodes

        # @var epsilon
        # The epsilon parameter of the PPO clip objective.
        self.epsilon = epsilon

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

    def get_policy_network(self):
        """
        Retrieve the policy network of the PPO agent.
        :return: the policy network.
        """
        network = ConvPolicyNetwork()
        network.train(self.training)
        network.to(self.device)
        return network

    def get_value_network(self):
        """
        Retrieve the value network of the PPO agent.
        :return: the value network.
        """
        network = ConvCriticNetwork()
        network.train(self.training)
        network.to(self.device)
        return network

    def step(self, obs: ObservationType) -> Tuple[ActionType, torch.Tensor]:
        """!
        Select the next action to perform in the environment.
        @param obs: the observation available to make the decision
        @return the next action to perform and the log-probability of the next action
        """
        probs = self.policy_net(obs)
        return np.random.choice(probs), probs.log()

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

    def rollouts(self, env, config):  # TODO typing
        """TODO"""

        # Collect the requested number of episodes.
        future_returns = []
        observations = []
        log_probs = []
        for i in range(self.n_episodes):

            # Retrieve the initial observation from the environment.
            obs, _ = env.reset()

            # Collect a single episode.
            rewards = []
            done = False
            while not done:

                # Collect the observations.
                observations.append(torch.unsqueeze(obs, dim=0))

                # Perform one step in the environment.
                action, log_prob = self.step(obs.to(self.device))
                old_obs = obs
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # TODO Add the experience to the replay buffer.
                # TODO self.buffer.append(Experience(old_obs, action, reward, done, obs))

                # TODO Perform one iteration of training (if needed).
                # TODO if self.current_step >= self.learning_starts:
                # TODO     self.learn()

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
        observations = torch.cat(observations).to(self.device)
        future_returns = torch.tensor(future_returns).to(self.device)
        return observations, log_probs, future_returns

    def value_loss(self, obs, rewards_to_go):
        """
        Compute the loss function of the value network.
        @param obs: the observation available to make the decision
        @param rewards_to_go: the rewards to go (sum of discounted future rewards)
        @return the loss function
        """
        loss_fc = MSELoss()
        loss = loss_fc(rewards_to_go, self.value_net(obs))
        return loss

    def policy_loss(self, obs, advantages, log_probs):
        """
        Compute the loss function of the policy network.
        @param obs: the observation available to make the decision
        @param advantages: the advantages of the observation-action pairs
        @param log_probs: the (old) log probabilities of the actions
        @return the loss function
        """
        current_log_probs = self.policy_net(obs).log()
        ratio = torch.exp(current_log_probs - log_probs)
        loss = torch.min(ratio * advantages, torch.clip(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages)
        return loss

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

            # Collect trajectories.
            obs, actions, log_probs, rewards, values, rewards_to_go = self.rollouts(env, config)

            self.n_training_steps = 16  # TODO move to constructor and hyperparameter tune
            for i in range(self.n_training_steps):

                # Compute the value loss.
                loss = self.value_loss(obs, rewards_to_go)

                # Perform one step of gradient descent on the value network with gradient clipping.
                self.value_optimizer.zero_grad()
                loss.mean().backward()
                for param in self.value_net.parameters():
                    param.grad.data.clamp_(-1, 1)
                self.value_optimizer.step()

                # Compute the advantages.
                advantages = rewards_to_go - values

                # Compute the policy loss.
                loss = self.policy_loss(obs, advantages, log_probs)

                # Perform one step of gradient descent on the policy network with gradient clipping.
                self.policy_optimizer.zero_grad()
                loss.mean().backward()
                for param in self.policy_net.parameters():
                    param.grad.data.clamp_(-1, 1)
                self.policy_optimizer.step()

            # TODO # Select an action.
            # TODO action = self.step(obs.to(self.device))

            # TODO # Execute the action in the environment.
            # TODO old_obs = obs
            # TODO obs, reward, terminated, truncated, _ = env.step(action)
            # TODO done = terminated or truncated

            # TODO # Add the experience to the replay buffer.
            # TODO self.buffer.append(Experience(old_obs, action, reward, done, obs))

            # TODO # Sample the replay buffer.
            # TODO obs, actions, rewards, done, next_obs = self.buffer.sample()
            # TODO # Compute the Q-value loss.
            # TODO loss = self.loss(obs, actions, rewards, done, next_obs)
            # TODO # Report the loss of the sampled transitions for prioritization.
            # TODO loss = self.buffer.report(loss)
            # TODO # Perform one step of gradient descent on the value network with
            # TODO # gradient clipping.
            # TODO self.policy_optimizer.zero_grad()
            # TODO loss.mean().backward()
            # TODO for param in self.value_net.parameters():
            # TODO     param.grad.data.clamp_(-1, 1)
            # TODO self.policy_optimizer.step()

            # Save the agent (if needed).
            if self.current_step % config["checkpoint_frequency"] == 0:
                self.save(f"model_{self.current_step}.pt")

            # TODO # Log the mean episodic reward in tensorboard (if needed).
            # TODO self.report(reward, done)
            # TODO if self.current_step % config["tensorboard_log_interval"] == 0:
            # TODO     self.log_performance_in_tensorboard()

            # TODO # Reset the environment when a trial ends.
            # TODO if done:
            # TODO     obs, _ = env.reset()

            # Increase the number of training steps done.
            self.current_step += 1  # TODO + n steps

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
                [self.policy_net], self.learning_rate, self.adam_eps, checkpoint, "policy_optimizer"
            )
            self.value_optimizer = get_adam_optimizer(
                [self.value_net], self.learning_rate, self.adam_eps, checkpoint, "value_optimizer"
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
            "epsilon": self.epsilon,
            "adam_eps": self.adam_eps,
            "n_actions": self.n_actions,
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
