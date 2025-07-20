import logging
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
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
        batch_size: int = 32,
        adam_eps: float = 1.5e-4,
        n_actions: int = 18,
        training: bool = True,
    ) -> None:
        """!
        Create a PPO agent.
        @param gamma: the discount factor
        @param learning_rate: the learning rate
        @param batch_size: the size of the batches sampled from the replay buffer
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

        # @var batch_size
        # Number of transitions sampled per learning update.
        self.batch_size = batch_size

        # @var adam_eps
        # Epsilon parameter for the Adam optimizer.
        self.adam_eps = adam_eps

        # @var policy_net
        # The policy network that predicts the distribution over actions.
        self.policy_net = self.get_policy_network()

        # @var critic_net
        # The critic network that predicts the Q-values of each action.
        self.critic_net = self.get_critic_network()

        # @var policy_optimizer
        # Adam optimizer for training the policy network.
        self.policy_optimizer = get_adam_optimizer(
            [self.policy_net],
            self.learning_rate,
            self.adam_eps,
        )

        # @var critic_optimizer
        # Adam optimizer for training the critic network.
        self.critic_optimizer = get_adam_optimizer(
            [self.critic_net],
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

    def get_critic_network(self):
        """
        Retrieve the critic network of the PPO agent.
        :return: the critic network.
        """
        network = ConvCriticNetwork()
        network.train(self.training)
        network.to(self.device)
        return network

    def step(self, obs: ObservationType) -> Tuple[ActionType, torch.Tensor]:
        """!
        Select the next action to perform in the environment.
        @param obs: the observation available to make the decision
        @return the next action to perform
        """
        probs = self.policy_net(obs)
        return np.random.choice(probs), probs.log()

    def train(self, env: Env) -> None:
        """! TODO
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
            self.buffer.append(Experience(old_obs, action, reward, done, obs))

            # Sample the replay buffer.
            obs, actions, rewards, done, next_obs = self.buffer.sample()
            # Compute the Q-value loss.
            loss = self.loss(obs, actions, rewards, done, next_obs)
            # Report the loss of the sampled transitions for prioritization.
            loss = self.buffer.report(loss)
            # Perform one step of gradient descent on the value network with
            # gradient clipping.
            self.policy_optimizer.zero_grad()
            loss.mean().backward()
            for param in self.value_net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.policy_optimizer.step()

            # Save the agent (if needed).
            if self.current_step % config["checkpoint_frequency"] == 0:
                self.save(f"model_{self.current_step}.pt")

            # Log the mean episodic reward in tensorboard (if needed).
            self.report(reward, done)
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
            self.critic_net = self.get_critic_network()
            safe_load_state_dict(self.critic_net, checkpoint, "critic_net")

            # Update the optimizers.
            self.policy_optimizer = get_adam_optimizer(
                [self.policy_net], self.learning_rate, self.adam_eps, checkpoint, "policy_optimizer"
            )
            self.critic_optimizer = get_adam_optimizer(
                [self.critic_net], self.learning_rate, self.adam_eps, checkpoint, "critic_optimizer"
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
            "batch_size": self.batch_size,
            "adam_eps": self.adam_eps,
            "n_actions": self.n_actions,
            "policy_net": self.policy_net.state_dict(),
            "critic_net": self.critic_net.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
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
