import logging
import os
from datetime import datetime
from enum import IntEnum
from functools import partial
from os.path import join

import torch
from torch.nn import SmoothL1Loss, MSELoss, CrossEntropyLoss

import benchmarks
from benchmarks.agents.AgentInterface import AgentInterface
from benchmarks.agents.memory.PrioritizedReplayBuffer import PrioritizedReplayBuffer
from benchmarks.agents.memory.ReplayBuffer import ReplayBuffer
from benchmarks.agents.memory.ReplayBufferInterface import Experience
from benchmarks.agents.networks.CategoricalDeepQNetworks import CategoricalDeepQNetwork, NoisyCategoricalDeepQNetwork
from benchmarks.agents.networks.DeepQNetworks import DeepQNetwork, NoisyDeepQNetwork
from torch import optim
import numpy as np

from benchmarks.agents.networks.DuelingDeepQNetworks import DuelingDeepQNetwork, NoisyDuelingDeepQNetwork


class LossType(IntEnum):
    """
    The loss functions supported by the DQN agent.
    """
    DQN_MSE = 0  # Q-value loss with Mean Square Error
    DQN_SL1 = 1  # Q-value loss with Smooth L1 norm
    DDQN_MSE = 2  # Double Q-value loss with Mean Square Error
    DDQN_SL1 = 3  # Double Q-value loss with Smooth L1 norm
    KL_DIVERGENCE = 4  # Categorical DQN loss


class ReplayType(IntEnum):
    """
    The replay buffer supported by the DQN agent.
    """
    DEFAULT = 0  # Standard replay buffer
    PRIORITIZED = 1  # Prioritized replay buffer


class NetworkType(IntEnum):
    """
    The networks supported by the DQN agent.
    """
    DEFAULT = 0  # Standard Deep Q-Network
    NOISY = 1  # Deep Q-Network with noisy linear layer
    DUELING = 2  # Dueling Deep Q-Network
    NOISY_DUELING = 3  # Dueling Deep Q-Network with noisy linear layer
    CATEGORICAL = 4  # Categorical Deep Q-network
    NOISY_CATEGORICAL = 5  # Categorical Deep Q-network with noisy linear layer


class DQN(AgentInterface):
    """
    Implement the Deep Q-Network agent from:
    Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves,
    Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al.
    Human-level control through deep reinforcement learning. nature, 2015.
    """

    def __init__(
        self, gamma=0.99, learning_rate=0.00001, buffer_size=1000000, batch_size=32, learning_starts=200000,
        target_update_interval=40000, adam_eps=1.5e-4, n_actions=18, n_atoms=1, v_min=None, v_max=None, training=True,
        replay_type=ReplayType.DEFAULT, loss_type=LossType.DQN_SL1, network_type=NetworkType.DEFAULT
    ):
        """
        Create a DQN agent.
        :param gamma: the discount factor
        :param learning_rate: the learning rate
        :param buffer_size: the size of the replay buffer
        :param batch_size: the size of the batches sampled from the replay buffer
        :param learning_starts: the step at which learning starts
        :param target_update_interval: number of training steps between two synchronization of the target
        :param adam_eps: the epsilon parameter of the Adam optimizer
        :param n_actions: the number of actions available to the agent
        :param n_atoms: the number of atoms used to approximate the distribution over returns
        :param v_min: the minimum amount of returns (only used for categorical DQN)
        :param v_max: the maximum amount of returns (only used for categorical DQN)
        :param training: True if the agent is being trained, False otherwise
        :param replay_type: the type of replay buffer
        :param loss_type: the loss to use during gradient descent
        :param network_type: the network architecture to use for the value and target networks
        """

        # Call the parent constructor.
        super().__init__()

        # A dictionary of loss functions.
        self.losses = {
            LossType.DQN_MSE: partial(self.q_learning_loss, loss_fc=MSELoss(reduction="none")),
            LossType.DQN_SL1: partial(self.q_learning_loss, loss_fc=SmoothL1Loss(reduction="none")),
            LossType.DDQN_MSE: partial(self.double_q_learning_loss, loss_fc=MSELoss(reduction="none")),
            LossType.DDQN_SL1: partial(self.double_q_learning_loss, loss_fc=SmoothL1Loss(reduction="none")),
            LossType.KL_DIVERGENCE: self.categorical_kl_divergence,
        }

        # A dictionary of replay buffers.
        self.buffers = {
            ReplayType.DEFAULT: ReplayBuffer,
            ReplayType.PRIORITIZED: PrioritizedReplayBuffer,
        }

        # A dictionary of value networks.
        self.networks = {
            NetworkType.DEFAULT: DeepQNetwork,
            NetworkType.NOISY: NoisyDeepQNetwork,
            NetworkType.DUELING: DuelingDeepQNetwork,
            NetworkType.NOISY_DUELING: NoisyDuelingDeepQNetwork,
            NetworkType.CATEGORICAL: CategoricalDeepQNetwork,
            NetworkType.NOISY_CATEGORICAL: NoisyCategoricalDeepQNetwork,
        }

        # Store the agent's parameters.
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.learning_starts = learning_starts
        self.adam_eps = adam_eps
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.n_actions = n_actions
        self.replay_type = replay_type
        self.loss_type = loss_type
        self.network_type = network_type
        self.training = training
        self.epsilon_schedule = [
            (0, 1), (self.learning_starts, 1), (1e6, 0.1), (10e6, 0.01)
        ]

        # Create the value network.
        self.value_net = self.networks[self.network_type](
            n_atoms=self.n_atoms, n_actions=self.n_actions, v_min=self.v_min, v_max=self.v_max
        )
        self.value_net.train(training)
        self.value_net.to(self.device)

        # Create the target network (copy value network's weights and avoid gradient computation).
        self.target_net = self.networks[self.network_type](
            n_atoms=self.n_atoms, n_actions=self.n_actions, v_min=self.v_min, v_max=self.v_max
        )
        self.target_net.train(training)
        self.target_net.to(self.device)
        self.update_target_network()
        for param in self.target_net.parameters():
            param.requires_grad = False

        # Create the optimizer and replay buffer.
        self.optimizer = optim.Adam(self.value_net.parameters(), lr=self.learning_rate, eps=self.adam_eps)
        self.buffer = self.buffers[self.replay_type](capacity=self.buffer_size)

    def update_target_network(self):
        """
        Synchronize the target with the value network.
        """
        self.target_net.load_state_dict(self.value_net.state_dict())

    def epsilon(self, current_step):
        """
        Compute the epsilon value at a given step.
        :param current_step: the step for which epsilon must be computed
        :return: the current epsilon
        """
        for i, (next_step, next_epsilon) in enumerate(self.epsilon_schedule):
            if next_step > current_step:
                previous_step, previous_epsilon = self.epsilon_schedule[i - 1]
                progress = (current_step - previous_step) / (next_step - previous_step)
                return progress * next_epsilon + (1 - progress) * previous_epsilon

        return self.epsilon_schedule[-1][1]

    def step(self, obs):
        """
        Select the next action to perform in the environment.
        :param obs: the observation available to make the decision
        :return: the next action to perform
        """
        if not self.training or np.random.random() > self.epsilon(self.current_step):
            return torch.argmax(self.value_net.q_values(obs), dim=1).item()
        return np.random.choice(self.n_actions)

    def train(self, env):
        """
        Train the agent in the gym environment passed as parameters
        :param env: the gym environment
        """

        # Retrieve the initial observation from the environment.
        obs, _ = env.reset()

        # Train the agent.
        config = benchmarks.config()
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

            # Perform one iteration of training (if needed).
            if len(self.buffer) >= self.learning_starts:
                self.learn()

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

    def learn(self):
        """
        Perform on step of gradient descent on the value network.
        """

        # Synchronize the target with the value network (if needed).
        if self.current_step % self.target_update_interval == 0:
            self.update_target_network()

        # Sample the replay buffer.
        obs, actions, rewards, done, next_obs = self.buffer.sample(self.batch_size)

        # Compute the Q-value loss.
        loss = self.losses[self.loss_type](obs, actions, rewards, done, next_obs)

        # Perform one step of gradient descent on the value network with gradient clipping.
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.value_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def q_learning_loss(self, obs, actions, rewards, done, next_obs, loss_fc):
        """
        Compute the loss of the Q-learning algorithm.
        :param obs: the observations at time t
        :param actions: the actions at time t
        :param rewards: the reward obtained when taking the actions while seeing the observations at time t
        :param done: whether the episodes ended
        :param next_obs: the observation at time t + 1
        :param loss_fc: the loss function to use to compare target and prediction
        :return: the Q-value loss
        """

        # Chose and evaluate the best actions using the target network.
        next_values = torch.max(self.target_net.q_values(next_obs), dim=1).values.detach()

        # Compute the Q-value loss.
        mask = torch.logical_not(done).float()
        y = rewards + mask * self.gamma * next_values
        x = self.value_net.q_values(obs)[range(self.batch_size), actions.squeeze()]
        loss = loss_fc(x, y)

        # Report the loss obtained for each sampled transition for potential prioritization.
        self.buffer.report(loss)

        # Report the total loss of the batch.
        return loss.mean()

    def double_q_learning_loss(self, obs, actions, rewards, done, next_obs, loss_fc):
        """
        Compute the loss of the double Q-learning algorithm.
        :param obs: the observations at time t
        :param actions: the actions at time t
        :param rewards: the reward obtained when taking the actions while seeing the observations at time t
        :param done: whether the episodes ended
        :param next_obs: the observation at time t + 1
        :param loss_fc: the loss function to use to compare target and prediction
        :return: the Q-value loss
        """

        # Chose the best actions according to the value network, and evaluate them using the target network.
        next_actions = torch.argmax(self.value_net.q_values(next_obs), dim=1).detach()
        next_values = self.target_net.q_values(next_obs)[range(self.batch_size), next_actions.squeeze()].detach()

        # Compute the Q-value loss.
        mask = torch.logical_not(done).float()
        y = rewards + mask * self.gamma * next_values
        x = self.value_net.q_values(obs)[range(self.batch_size), actions.squeeze()]
        loss = loss_fc(x, y)

        # Report the loss obtained for each sampled transition for potential prioritization.
        self.buffer.report(loss)

        # Report the total loss of the batch.
        return loss.mean()

    def categorical_kl_divergence(self, obs, actions, rewards, done, next_obs):
        """
        Compute the loss of the categorical algorithm.
        :param obs: the observations at time t
        :param actions: the actions at time t
        :param rewards: the reward obtained when taking the actions while seeing the observations at time t
        :param done: whether the episodes ended
        :param next_obs: the observation at time t + 1
        :return: the categorical loss
        """

        # Compute the best actions at time t + 1.
        next_atoms, next_probs, _ = self.value_net(next_obs)
        next_q_values = (next_atoms * next_probs).sum(dim=1)
        next_actions = torch.argmax(next_q_values, dim=1)

        # Retrieve the atoms and probabilities corresponding to the best actions at time t + 1.
        next_atoms = next_atoms[range(self.batch_size), :, next_actions.squeeze()]
        next_probs = next_probs[range(self.batch_size), :, next_actions.squeeze()]

        # Compute the new atom positions using the Bellman update.
        next_atoms = rewards.unsqueeze(dim=1).repeat(1, self.n_atoms) + self.gamma * next_atoms
        next_atoms = torch.clamp(next_atoms, self.v_min, self.v_max)

        # Compute the projected distribution over returns.
        target_probs = torch.zeros_like(next_probs)
        for j in range(self.n_atoms):
            atom = (next_atoms[:, j] - self.v_min) / self.value_net.delta_z
            lower = torch.floor(atom).int()
            upper = torch.ceil(atom).int()
            target_probs[range(self.batch_size), lower] += next_probs[range(self.batch_size), j] * (upper - atom)
            mask = torch.logical_not(torch.eq(lower, upper))
            target_probs[range(self.batch_size), upper] += mask * next_probs[range(self.batch_size), j] * (atom - lower)

        # Compute the predicted return log-probabilities.
        _, _, log_probs = self.value_net(obs)
        log_probs = log_probs[range(self.batch_size), :, actions.squeeze()]

        # Compute the categorical loss.
        loss_fc = CrossEntropyLoss(reduction="none")
        loss = loss_fc(log_probs, target_probs.detach())

        # Report the loss obtained for each sampled transition for potential prioritization.
        self.buffer.report(loss)

        # Report the total loss of the batch.
        return loss.mean()

    def load(self, checkpoint_name=None):
        """
        Load an agent from the filesystem.
        :param checkpoint_name: the name of the checkpoint to load
        """

        # Retrieve the full checkpoint path.
        if checkpoint_name is None:
            checkpoint_path = self.get_latest_checkpoint()
        else:
            checkpoint_path = join(os.environ["CHECKPOINT_DIRECTORY"], checkpoint_name)

        # Check if the checkpoint can be loaded.
        if checkpoint_path is None:
            return 
        
        # Load the checkpoint from the file system.
        logging.info("Loading agent from the following file: " + checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Update the agent's parameters using the checkpoint.
        self.gamma = checkpoint["gamma"]
        self.learning_rate = checkpoint["learning_rate"]
        self.buffer_size = checkpoint["buffer_size"]
        self.batch_size = checkpoint["batch_size"]
        self.target_update_interval = checkpoint["target_update_interval"]
        self.learning_starts = checkpoint["learning_starts"]
        self.adam_eps = checkpoint["adam_eps"]
        self.n_atoms = checkpoint["n_atoms"]
        self.v_min = checkpoint["v_min"]
        self.v_max = checkpoint["v_max"]
        self.n_actions = checkpoint["n_actions"]
        self.replay_type = checkpoint["replay_type"]
        self.loss_type = checkpoint["loss_type"]
        self.network_type = checkpoint["network_type"]
        self.training = checkpoint["training"]
        self.epsilon_schedule = checkpoint["epsilon_schedule"]
        self.current_step = checkpoint["current_step"]

        # Update the agent's networks using the checkpoint.
        self.value_net = self.networks[self.network_type](
            n_atoms=self.n_atoms, n_actions=self.n_actions, v_min=self.v_min, v_max=self.v_max
        )
        self.value_net.load_state_dict(checkpoint[f"value_net"])
        self.value_net.train(self.training)
        self.value_net.to(self.device)

        self.target_net = self.networks[self.network_type](
            n_atoms=self.n_atoms, n_actions=self.n_actions, v_min=self.v_min, v_max=self.v_max
        )
        self.target_net.load_state_dict(checkpoint[f"target_net"])
        self.target_net.train(self.training)
        self.target_net.to(self.device)
        for param in self.target_net.parameters():
            param.requires_grad = False

        # Update the optimizer and replay buffer.
        self.optimizer = optim.Adam(self.value_net.parameters(), lr=self.learning_starts, eps=self.adam_eps)
        self.buffer = self.buffers[self.replay_type](capacity=self.buffer_size)

    def save(self, checkpoint_name):
        """
        Save the agent on the filesystem.
        :param checkpoint_name: the name of the checkpoint in which to save the agent
        """
        
        # Create the checkpoint directory and file, if they do not exist.
        checkpoint_path = join(os.environ["CHECKPOINT_DIRECTORY"], checkpoint_name)
        self.create_directory_and_file(checkpoint_path)

        # Save the model.
        torch.save({
            "gamma": self.gamma,
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "batch_size": self.batch_size,
            "target_update_interval": self.target_update_interval,
            "learning_starts": self.learning_starts,
            "adam_eps": self.adam_eps,
            "n_atoms": self.n_atoms,
            "v_min": self.v_min,
            "v_max": self.v_max,
            "n_actions": self.n_actions,
            "replay_type": self.replay_type,
            "loss_type": self.loss_type,
            "network_type": self.network_type,
            "training": self.training,
            "epsilon_schedule": self.epsilon_schedule,
            "current_step": self.current_step,
            "value_net": self.value_net.state_dict(),
            "target_net": self.target_net.state_dict(),
        }, checkpoint_path)
