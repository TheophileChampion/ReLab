import logging
import math
from datetime import datetime
from functools import partial
from typing import Optional, Tuple, List, SupportsFloat, Dict, Any
from torch.nn import MSELoss

from relab.cpp.agents.memory import Experience

import relab
import torch
from torch import Tensor
from torch import nn
from torch.distributions.categorical import Categorical
from gymnasium import Env
from relab.agents.AgentInterface import AgentInterface, ReplayType
from relab.agents.networks.CriticNetwork import ConvCriticNetwork

from relab.agents.networks.PolicyNetwork import ConvPolicyNetwork
from relab.helpers.Serialization import get_adam_optimizer, safe_load_state_dict
from relab.helpers.Typing import (
    ActionType,
    AttributeNames,
    Checkpoint,
    Config,
    ObservationType,
    ConfigInfo,
    Loss,
)

class A2C(AgentInterface):
    """!
    @brief Implements an advantage actor critic (A2C) agent.

    @details
    This implementation is based on the paper:

    <b>Asynchronous Methods for Deep Reinforcement Learning</b>,
    published in arXiv, 2016.

    Authors:
    - Volodymyr Mnih
    - Adria Puigdomenech Badia
    - Mehdi Mirza
    - Alex Graves
    - Timothy P. Lillicrap
    - Tim Harley
    - David Silver
    - Koray Kavukcuoglu

    The paper discusses a number of asynchronous reinforcement learning algorithms.
    This class implements a synchronous version of the A3C algorithm.
    """

    def __init__(
        self,
        gamma: float = 0.99,
        learning_rate: float = 0.00001,
        n_episodes: int = 10,
        adam_eps: float = 1.5e-4,
        n_actions: int = 18,
        training: bool = True,
        buffer_size: int = 1000000,
        batch_size: int = 32,
        replay_type: ReplayType = ReplayType.DEFAULT,
        omega: float = 1.0,
        omega_is: float = 1.0,
        n_steps: int = 1,
        target_update_interval: int = 10000,
        learning_starts: int = 10000,
    ) -> None:
        """!
        Create an advantage actor critic agent.
        @param gamma: the discount factor
        @param learning_rate: the learning rate
        @param n_episodes: the number of episodes run per (policy) learning update
        @param adam_eps: the epsilon parameter of the Adam optimizer
        @param n_actions: the number of actions available to the agent
        @param training: True if the agent is being trained, False otherwise
        @param buffer_size: the size of the replay buffer
        @param batch_size: the size of the batches sampled from the replay buffer
        @param n_steps: the number of steps for which rewards are accumulated in multistep Q-learning
        @param omega: the prioritization exponent
        @param omega_is: the important sampling exponent
        @param replay_type: the type of replay buffer
        @param target_update_interval: number of training steps between two synchronization of the target
        @param learning_starts: the step at which learning starts
        """

        # Call the parent constructor.
        buffer = partial(
            self.get_replay_buffer,
            buffer_size,
            batch_size,
            replay_type,
            omega,
            omega_is,
            n_steps,
            gamma,
        )
        super().__init__(get_buffer=buffer, n_actions=n_actions, training=training)

        # @var gamma
        # Discount factor for future rewards (between 0 and 1).
        self.gamma = gamma

        # @var learning_rate
        # Learning rate for the optimizers.
        self.learning_rate = learning_rate

        # @var n_episodes
        # Number of episodes run per learning update of the policy network.
        self.n_episodes = n_episodes

        # @var adam_eps
        # Epsilon parameter for the Adam optimizers.
        self.adam_eps = adam_eps

        # @var buffer_size
        # Maximum number of transitions stored in the replay buffer.
        self.buffer_size = buffer_size

        # @var batch_size
        # Number of transitions sampled per learning update of the value network.
        self.batch_size = batch_size

        # @var n_steps
        # Number of steps for multi-step learning.
        self.n_steps = n_steps

        # @var omega
        # Exponent for prioritization in the replay buffer.
        self.omega = omega

        # @var omega_is
        # Exponent for importance sampling correction.
        self.omega_is = omega_is

        # @var replay_type
        # Type of experience replay buffer being used.
        self.replay_type = replay_type

        # @var target_update_interval
        # Number of training steps between target network updates.
        self.target_update_interval = target_update_interval

        # @var learning_starts
        # Step count at which learning begins.
        self.learning_starts = learning_starts

        # @var value_net
        # The value network that predicts the value of the current state.
        self.value_net = self.get_value_network()

        # @var target_net
        # The target network, which is a copy of the value network synchronized
        # periodically.
        self.target_net = self.get_value_network()
        self.update_target_network()
        for param in self.target_net.parameters():
            param.requires_grad = False

        # @var policy_net
        # The policy network that predicts the distribution over actions.
        self.policy_net = self.get_policy_network()

        # @var optimizer
        # Adam optimizer for training the value network.
        self.optimizer = get_adam_optimizer(
            [self.value_net], self.learning_rate, self.adam_eps,
        )

        # @var policy_optimizer
        # Adam optimizer for training the policy network.
        self.policy_optimizer = get_adam_optimizer(
            [self.policy_net], self.learning_rate, self.adam_eps,
        )

    def get_policy_network(self) -> nn.Module:
        """
        Retrieve the policy network of the A2C agent.
        @return the policy network
        """
        network = ConvPolicyNetwork()
        network.train(self.training)
        network.to(self.device)
        return network

    def get_value_network(self):
        """
        Retrieve the critic network of the A2C agent.
        :return: the critic network.
        """
        network = ConvCriticNetwork(n_outputs=1)
        network.train(self.training)
        network.to(self.device)
        return network

    def update_target_network(self) -> None:
        """!
        Synchronize the target with the value network.
        """
        self.target_net.load_state_dict(self.value_net.state_dict())

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

    def rollouts(self, env: Env, config: ConfigInfo) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Run rollouts to collect a batch of data.
        @param env: the environment in which the episodes are run
        @param config: the training configuration
        @return a tuple (observations, log-probability of actions, future_returns)
        """

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

                # Add the experience to the replay buffer.
                self.buffer.append(Experience(old_obs, action, reward, done, obs))

                # Perform one iteration of training (if needed).
                if self.current_step >= self.learning_starts:
                    self.learn()

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

    def learn(self) -> Optional[Dict[str, Any]]:
        """!
        Perform one step of gradient descent on the value network.
        """

        # Synchronize the target with the value network (if needed).
        if self.current_step % self.target_update_interval == 0:
            self.update_target_network()

        # Sample the replay buffer.
        obs, actions, rewards, done, next_obs = self.buffer.sample()

        # Compute the Q-value loss.
        loss = self.compute_value_loss(
            obs, actions, rewards, done, next_obs,
            loss_fc=MSELoss(reduction="none")
        )

        # Report the loss of the sampled transitions for prioritization.
        loss = self.buffer.report(loss)

        # Perform one step of gradient descent on the value network with
        # gradient clipping.
        self.optimizer.zero_grad()
        loss.mean().backward()
        for param in self.value_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return None

    def compute_value_loss(
        self,
        obs: Tensor,
        actions: Tensor,
        rewards: Tensor,
        done: Tensor,
        next_obs: Tensor,
        loss_fc: Loss,
    ) -> Tensor:
        """!
        Compute the loss of the standard or double Q-learning algorithm.
        @param obs: the observations at time t
        @param actions: the actions at time t
        @param rewards: the reward obtained when taking the actions while seeing the observations at time t
        @param done: whether the episodes ended
        @param next_obs: the observation at time t + 1
        @param loss_fc: the loss function to use to compare target and prediction
        @return the Q-value loss
        """

        # Chose and evaluate the best actions using the target network.
        next_values = self.target_net(next_obs)
        next_values = torch.max(next_values, dim=1).values
        next_values = next_values.detach()

        # Compute the Q-value loss.
        mask = torch.logical_not(done).float()
        y = rewards + mask * math.pow(self.gamma, self.n_steps) * next_values
        return loss_fc(torch.squeeze(self.value_net(obs)), y)

    def compute_policy_loss(
        self,
        observations: Tensor,
        log_probs: Tensor,
        future_returns: Tensor
    ) -> Tensor:
        """
        Compute the policy gradient loss.
        :param observations: the observations used to select the actions
        :param log_probs: the log-probability of the selected actions
        :param future_returns: the future returns obtained by following the current policy
        :return: the loss
        """
        return -log_probs * (future_returns - self.value_net(observations).detach())

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
            observations, log_probs, future_returns = self.rollouts(env, config)

            # Perform one step of gradient descent to train the policy network.
            self.policy_optimizer.zero_grad()
            loss = self.compute_policy_loss(observations, log_probs, future_returns)
            loss.mean().backward()
            self.policy_optimizer.step()

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
            self.value_net = self.get_value_network()
            safe_load_state_dict(self.value_net, checkpoint, "value_net")

            self.target_net = self.get_value_network()
            safe_load_state_dict(self.target_net, checkpoint, "target_net")
            for param in self.target_net.parameters():
                param.requires_grad = False

            self.policy_net = self.get_policy_network()
            safe_load_state_dict(self.policy_net, checkpoint, "policy_net")

            # Update the optimizers.
            self.optimizer = get_adam_optimizer(
                [self.value_net], self.learning_rate, self.adam_eps, checkpoint
            )
            self.policy_optimizer = get_adam_optimizer(
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
            "value_net": self.value_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "policy_net": self.policy_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "buffer_size": self.buffer_size,
            "batch_size": self.batch_size,
            "replay_type": self.replay_type,
            "omega": self.omega,
            "omega_is": self.omega_is,
            "n_steps": self.n_steps,
            "target_update_interval": self.target_update_interval,
            "learning_starts": self.learning_starts,
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
