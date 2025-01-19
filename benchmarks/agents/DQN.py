import logging
import math
import os
from datetime import datetime
from enum import IntEnum
from functools import partial
from os.path import join

import torch
from torch.nn import SmoothL1Loss, MSELoss, CrossEntropyLoss, HuberLoss

import benchmarks
from benchmarks.agents.AgentInterface import AgentInterface, ReplayType
from benchmarks.agents.memory.ReplayBuffer import Experience
from benchmarks.agents.networks.CategoricalDeepQNetworks import CategoricalDeepQNetwork, NoisyCategoricalDeepQNetwork
from benchmarks.agents.networks.DeepQNetworks import DeepQNetwork, NoisyDeepQNetwork
from torch import optim
import numpy as np

from benchmarks.agents.networks.DuelingDeepQNetworks import DuelingDeepQNetwork, NoisyDuelingDeepQNetwork
from benchmarks.agents.networks.QuantileDeepQNetworks import QuantileDeepQNetwork, ImplicitQuantileNetwork
from benchmarks.agents.networks.RainbowDeepQNetwork import RainbowDeepQNetwork, RainbowImplicitQuantileNetwork
from benchmarks.agents.schedule.PiecewiseLinearSchedule import PiecewiseLinearSchedule
from benchmarks.helpers.FileSystem import FileSystem


class LossType(IntEnum):
    """
    The loss functions supported by the DQN agent.
    """
    DQN_MSE = 0  # Q-value loss with Mean Square Error
    DQN_SL1 = 1  # Q-value loss with Smooth L1 norm
    DDQN_MSE = 2  # Double Q-value loss with Mean Square Error
    DDQN_SL1 = 3  # Double Q-value loss with Smooth L1 norm
    KL_DIVERGENCE = 4  # Categorical DQN loss
    QUANTILE = 5  # Huber quantile loss
    IMPLICIT_QUANTILE = 6  # Implicit quantile loss
    RAINBOW = 7  # Rainbow loss
    RAINBOW_IQN = 8  # Rainbow IQN loss


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
    QUANTILE = 6  # Quantile Deep Q-network
    IMPLICIT_QUANTILE = 7  # Implicit quantile network
    RAINBOW = 8  # Rainbow Deep Q-network
    RAINBOW_IQN = 9  # Rainbow Implicit Q-network


class DQN(AgentInterface):
    """
    Implement the Deep Q-Network agent from:
    Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves,
    Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al.
    Human-level control through deep reinforcement learning. nature, 2015.
    """

    def __init__(
        self, gamma=0.99, learning_rate=0.00001, buffer_size=1000000, batch_size=32, learning_starts=200000, kappa=None,
        target_update_interval=40000, adam_eps=1.5e-4, n_actions=18, n_atoms=None, v_min=None, v_max=None, n_steps=1,
        training=True, replay_type=ReplayType.DEFAULT, loss_type=LossType.DQN_SL1, network_type=NetworkType.DEFAULT,
        omega=1.0, omega_is=1.0, epsilon_schedule=None
    ):
        """
        Create a DQN agent.
        :param gamma: the discount factor
        :param learning_rate: the learning rate
        :param buffer_size: the size of the replay buffer
        :param batch_size: the size of the batches sampled from the replay buffer
        :param learning_starts: the step at which learning starts
        :param kappa: the kappa parameter of the quantile Huber loss see Equation (10) in QR-DQN paper
        :param target_update_interval: number of training steps between two synchronization of the target
        :param adam_eps: the epsilon parameter of the Adam optimizer
        :param n_actions: the number of actions available to the agent
        :param n_atoms: the number of atoms used to approximate the distribution over returns
        :param v_min: the minimum amount of returns (only used for distributional DQN)
        :param v_max: the maximum amount of returns (only used for distributional DQN)
        :param training: True if the agent is being trained, False otherwise
        :param n_steps: the number of steps for which rewards are accumulated in multistep Q-learning
        :param omega: the prioritization exponent
        :param omega_is: the important sampling exponent
        :param replay_type: the type of replay buffer
        :param loss_type: the loss to use during gradient descent
        :param network_type: the network architecture to use for the value and target networks
        :param epsilon_schedule: the schedule for the exploration parameter epsilon as a list of tuple, i.e.,
            [(step_1, value_1), (step_2, value_2), ..., (step_n, value_n)]
        """

        # Call the parent constructor.
        super().__init__(training=training)

        # Store the agent's parameters.
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.learning_starts = learning_starts
        self.kappa = kappa
        self.adam_eps = adam_eps
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.n_actions = n_actions
        self.n_steps = n_steps
        self.omega = omega
        self.omega_is = omega_is
        self.replay_type = replay_type
        self.loss_type = loss_type
        self.network_type = network_type
        self.training = training
        self.epsilon_schedule = [
            (0, 1), (self.learning_starts, 1), (1e6, 0.1), (10e6, 0.01)
        ] if epsilon_schedule is None else epsilon_schedule
        self.epsilon = PiecewiseLinearSchedule(self.epsilon_schedule)

        # The loss function, value network, and replay buffer.
        self.loss = self.get_loss(self.loss_type)
        network = self.get_value_network(self.network_type)
        replay_buffer = self.get_replay_buffer(self.replay_type, self.omega, self.omega_is, self.n_steps, self.gamma)

        # Create the value network.
        self.value_net = network()
        self.value_net.train(training)
        self.value_net.to(self.device)

        # Create the target network (copy value network's weights and avoid gradient computation).
        self.target_net = network()
        self.target_net.train(training)
        self.target_net.to(self.device)
        self.update_target_network()
        for param in self.target_net.parameters():
            param.requires_grad = False

        # Create the optimizer and replay buffer.
        self.optimizer = optim.Adam(self.value_net.parameters(), lr=self.learning_rate, eps=self.adam_eps)
        self.buffer = replay_buffer(capacity=self.buffer_size, batch_size=self.batch_size)

    def get_loss(self, loss_type):
        """
        Retrieve the loss requested as parameters.
        :param loss_type: the loss to use during gradient descent
        :return: the loss
        """
        return {
            LossType.DQN_MSE: partial(self.q_learning_loss, loss_fc=MSELoss(reduction="none")),
            LossType.DQN_SL1: partial(self.q_learning_loss, loss_fc=SmoothL1Loss(reduction="none")),
            LossType.DDQN_MSE: partial(self.q_learning_loss, loss_fc=MSELoss(reduction="none"), double_ql=True),
            LossType.DDQN_SL1: partial(self.q_learning_loss, loss_fc=SmoothL1Loss(reduction="none"), double_ql=True),
            LossType.KL_DIVERGENCE: self.categorical_kl_divergence,
            LossType.QUANTILE: partial(self.quantile_loss, kappa=self.kappa),
            LossType.IMPLICIT_QUANTILE: partial(self.implicit_quantile_loss, kappa=self.kappa),
            LossType.RAINBOW: self.rainbow_loss,
            LossType.RAINBOW_IQN: partial(self.rainbow_iqn_loss, kappa=self.kappa),
        }[loss_type]

    def get_value_network(self, network_type):
        """
        Retrieve the constructor of the value network requested as parameters.
        :param network_type: the network architecture to use for the value and target networks
        :return: the constructor of the value network
        """
        return {
            NetworkType.DEFAULT: partial(DeepQNetwork, n_actions=self.n_actions),
            NetworkType.NOISY: partial(NoisyDeepQNetwork, n_actions=self.n_actions),
            NetworkType.DUELING: partial(DuelingDeepQNetwork, n_actions=self.n_actions),
            NetworkType.NOISY_DUELING: partial(NoisyDuelingDeepQNetwork, n_actions=self.n_actions),
            NetworkType.CATEGORICAL: partial(CategoricalDeepQNetwork, n_actions=self.n_actions, n_atoms=self.n_atoms, v_min=self.v_min, v_max=self.v_max),
            NetworkType.NOISY_CATEGORICAL: partial(NoisyCategoricalDeepQNetwork, n_actions=self.n_actions, n_atoms=self.n_atoms, v_min=self.v_min, v_max=self.v_max),
            NetworkType.QUANTILE: partial(QuantileDeepQNetwork, n_actions=self.n_actions, n_atoms=self.n_atoms),
            NetworkType.IMPLICIT_QUANTILE: partial(ImplicitQuantileNetwork, n_actions=self.n_actions),
            NetworkType.RAINBOW: partial(RainbowDeepQNetwork, n_actions=self.n_actions, n_atoms=self.n_atoms, v_min=self.v_min, v_max=self.v_max),
            NetworkType.RAINBOW_IQN: partial(RainbowImplicitQuantileNetwork, n_actions=self.n_actions),
        }[network_type]

    def update_target_network(self):
        """
        Synchronize the target with the value network.
        """
        self.target_net.load_state_dict(self.value_net.state_dict())

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
            if self.current_step >= self.learning_starts:
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
        Perform one step of gradient descent on the value network.
        """

        # Synchronize the target with the value network (if needed).
        if self.current_step % self.target_update_interval == 0:
            self.update_target_network()

        # Sample the replay buffer.
        obs, actions, rewards, done, next_obs = self.buffer.sample()

        # Compute the Q-value loss.
        loss = self.loss(obs, actions, rewards, done, next_obs)

        # Report the loss obtained for each sampled transition for potential prioritization.
        loss = self.buffer.report(loss)

        # Perform one step of gradient descent on the value network with gradient clipping.
        self.optimizer.zero_grad()
        loss.mean().backward()
        for param in self.value_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def q_learning_loss(self, obs, actions, rewards, done, next_obs, loss_fc, double_ql=False):
        """
        Compute the loss of the standard or double Q-learning algorithm.
        :param obs: the observations at time t
        :param actions: the actions at time t
        :param rewards: the reward obtained when taking the actions while seeing the observations at time t
        :param done: whether the episodes ended
        :param next_obs: the observation at time t + 1
        :param loss_fc: the loss function to use to compare target and prediction
        :param double_ql: False for standard Q-learning, True for Double Q-learning
        :return: the Q-value loss
        """

        if double_ql is True:
            # Chose the best actions according to the value network, and evaluate them using the target network.
            next_actions = torch.argmax(self.value_net.q_values(next_obs), dim=1).detach()
            next_values = self.target_net.q_values(next_obs)[range(self.batch_size), next_actions.squeeze()].detach()
        else:
            # Chose and evaluate the best actions using the target network.
            next_values = torch.max(self.target_net.q_values(next_obs), dim=1).values.detach()

        # Compute the Q-value loss.
        mask = torch.logical_not(done).float()
        y = rewards + mask * math.pow(self.gamma, self.n_steps) * next_values
        x = self.value_net.q_values(obs)[range(self.batch_size), actions.squeeze()]
        loss = loss_fc(x, y)
        return loss

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
        next_atoms, next_probs, _ = self.target_net(next_obs)
        next_q_values = (next_atoms * next_probs).sum(dim=1)
        next_actions = torch.argmax(next_q_values, dim=1)

        # Retrieve the atoms and probabilities corresponding to the best actions at time t + 1.
        next_atoms = next_atoms[range(self.batch_size), :, next_actions.squeeze()]
        next_probs = next_probs[range(self.batch_size), :, next_actions.squeeze()]

        # Compute the new atom positions using the Bellman update.
        next_atoms = rewards.unsqueeze(dim=1).repeat(1, self.n_atoms) + math.pow(self.gamma, self.n_steps) * next_atoms
        next_atoms = torch.clamp(next_atoms, self.v_min, self.v_max)

        # Compute the projected distribution over returns.
        target_probs = torch.zeros_like(next_probs)
        for j in range(self.n_atoms):
            atom = (next_atoms[:, j] - self.v_min) / self.target_net.delta_z
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
        return loss

    def rainbow_loss(self, obs, actions, rewards, done, next_obs):
        """
        Compute the loss of the rainbow DQN.
        :param obs: the observations at time t
        :param actions: the actions at time t
        :param rewards: the reward obtained when taking the actions while seeing the observations at time t
        :param done: whether the episodes ended
        :param next_obs: the observation at time t + 1
        :return: the rainbow loss
        """

        # Compute the best actions at time t + 1 using the value network.
        next_actions = torch.argmax(self.value_net.q_values(next_obs), dim=1).detach()

        # Retrieve the atoms and probabilities corresponding to the best actions at time t + 1.
        next_atoms, next_probs, _ = self.target_net(next_obs)
        next_atoms = next_atoms[range(self.batch_size), :, next_actions.squeeze()]
        next_probs = next_probs[range(self.batch_size), :, next_actions.squeeze()]

        # Compute the new atom positions using the Bellman update.
        next_atoms = rewards.unsqueeze(dim=1).repeat(1, self.n_atoms) + math.pow(self.gamma, self.n_steps) * next_atoms
        next_atoms = torch.clamp(next_atoms, self.v_min, self.v_max)

        # Compute the projected distribution over returns.
        target_probs = torch.zeros_like(next_probs)
        for j in range(self.n_atoms):
            atom = (next_atoms[:, j] - self.v_min) / self.target_net.delta_z
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
        return loss

    def quantile_loss(self, obs, actions, rewards, done, next_obs, kappa=1.0):
        """
        Compute the loss of the quantile regression algorithm.
        :param obs: the observations at time t
        :param actions: the actions at time t
        :param rewards: the reward obtained when taking the actions while seeing the observations at time t
        :param done: whether the episodes ended
        :param next_obs: the observation at time t + 1
        :param kappa: the kappa parameter of the quantile Huber loss see Equation (10) in QR-DQN paper
        :return: the categorical loss
        """

        # Compute the best actions at time t + 1.
        next_atoms = self.target_net(next_obs)
        next_q_values = next_atoms.sum(dim=1) / self.n_atoms
        next_actions = torch.argmax(next_q_values, dim=1)

        # Compute the new atom positions using the Bellman update.
        next_atoms = next_atoms[range(self.batch_size), :, next_actions.squeeze()]
        next_atoms = rewards.unsqueeze(dim=1).repeat(1, self.n_atoms) + math.pow(self.gamma, self.n_steps) * next_atoms

        # Compute the predicted atoms (canonical return).
        atoms = self.value_net(obs)
        atoms = atoms[range(self.batch_size), :, actions.squeeze()]

        # Compute the quantile Huber loss.
        huber_loss = HuberLoss(reduction="none", delta=kappa)
        loss = torch.zeros([self.batch_size]).to(self.device)
        for i in range(self.n_atoms):
            tau = (i + 0.5) / self.n_atoms
            for j in range(self.n_atoms):
                next_atom_j = next_atoms[:, j]
                atom_i = atoms[:, i]
                mask = torch.where(next_atom_j - atom_i < 0, 1.0, 0.0)
                loss += torch.abs(tau - mask).to(self.device) * huber_loss(next_atom_j, atom_i) / kappa
        loss /= self.n_atoms
        return loss

    def implicit_quantile_loss(self, obs, actions, rewards, done, next_obs, kappa=1.0):
        """
        Compute the loss of the quantile regression algorithm.
        :param obs: the observations at time t
        :param actions: the actions at time t
        :param rewards: the reward obtained when taking the actions while seeing the observations at time t
        :param done: whether the episodes ended
        :param next_obs: the observation at time t + 1
        :param kappa: the kappa parameter of the quantile Huber loss see Equation (10) in QR-DQN paper
        :return: the categorical loss
        """

        # Compute the best actions at time t + 1.
        next_q_values = self.target_net.q_values(next_obs)
        next_actions = torch.argmax(next_q_values, dim=1)

        # Compute the new atom positions using the Bellman update.
        next_atoms, _ = self.target_net(next_obs, n_samples=self.n_atoms, invalidate_cache=False)
        next_atoms = next_atoms[range(self.batch_size), :, next_actions.squeeze()]
        next_atoms = rewards.unsqueeze(dim=1).repeat(1, self.n_atoms) + math.pow(self.gamma, self.n_steps) * next_atoms

        # Compute the predicted atoms (canonical return).
        atoms, taus = self.value_net(obs, n_samples=self.n_atoms)
        atoms = atoms[range(self.batch_size), :, actions.squeeze()]

        # Compute the quantile Huber loss.
        huber_loss = HuberLoss(reduction="none", delta=kappa)
        loss = torch.zeros([self.batch_size]).to(self.device)
        for i in range(self.n_atoms):
            atom_i = atoms[:, i]
            for j in range(self.n_atoms):
                next_atom_j = next_atoms[:, j]
                mask = torch.where(next_atom_j - atom_i < 0, 1.0, 0.0)
                loss += torch.abs(taus[:, i] - mask).to(self.device) * huber_loss(next_atom_j, atom_i) / kappa
        loss /= self.n_atoms
        return loss

    def rainbow_iqn_loss(self, obs, actions, rewards, done, next_obs, kappa=1.0):
        """
        Compute the loss of the rainbow IQN.
        :param obs: the observations at time t
        :param actions: the actions at time t
        :param rewards: the reward obtained when taking the actions while seeing the observations at time t
        :param done: whether the episodes ended
        :param next_obs: the observation at time t + 1
        :param kappa: the kappa parameter of the quantile Huber loss see Equation (3) in IQN paper
        :return: the rainbow IQN loss
        """

        # Compute the best actions at time t + 1 using the value network.
        next_actions = torch.argmax(self.value_net.q_values(next_obs), dim=1).detach()

        # Compute the new atom positions using the Bellman update.
        next_atoms, _ = self.target_net(next_obs, n_samples=self.n_atoms)
        next_atoms = next_atoms[range(self.batch_size), :, next_actions.squeeze()]
        next_atoms = rewards.unsqueeze(dim=1).repeat(1, self.n_atoms) + math.pow(self.gamma, self.n_steps) * next_atoms

        # Compute the predicted atoms (canonical return) at time t.
        atoms, taus = self.value_net(obs, n_samples=self.n_atoms)
        atoms = atoms[range(self.batch_size), :, actions.squeeze()]

        # Compute the quantile Huber loss.
        huber_loss = HuberLoss(reduction="none", delta=kappa)
        loss = torch.zeros([self.batch_size]).to(self.device)
        for i in range(self.n_atoms):
            atom_i = atoms[:, i]
            for j in range(self.n_atoms):
                next_atom_j = next_atoms[:, j]
                mask = torch.where(next_atom_j - atom_i < 0, 1.0, 0.0)
                loss += torch.abs(taus[:, i] - mask).to(self.device) * huber_loss(next_atom_j, atom_i) / kappa
        loss /= self.n_atoms
        return loss

    def load(self, checkpoint_name=None, buffer_checkpoint_name=None):
        """
        Load an agent from the filesystem.
        :param checkpoint_name: the name of the agent checkpoint to load
        :param buffer_checkpoint_name: the name of the replay buffer checkpoint to load (None for default name)
        """

        # Retrieve the full agent checkpoint path.
        if checkpoint_name is None:
            checkpoint_path = self.get_latest_checkpoint()
        else:
            checkpoint_path = join(os.environ["CHECKPOINT_DIRECTORY"], checkpoint_name)

        # Check if the checkpoint can be loaded.
        if checkpoint_path is None:
            logging.info("Could not load the agent from the file system.")
            return
        
        # Load the checkpoint from the file system.
        logging.info("Loading agent from the following file: " + checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Update the agent's parameters using the checkpoint.
        self.gamma = self.safe_load(checkpoint, "gamma")
        self.learning_rate = self.safe_load(checkpoint, "learning_rate")
        self.buffer_size = self.safe_load(checkpoint, "buffer_size")
        self.batch_size = self.safe_load(checkpoint, "batch_size")
        self.target_update_interval = self.safe_load(checkpoint, "target_update_interval")
        self.learning_starts = self.safe_load(checkpoint, "learning_starts")
        self.kappa = self.safe_load(checkpoint, "kappa")
        self.adam_eps = self.safe_load(checkpoint, "adam_eps")
        self.n_atoms = self.safe_load(checkpoint, "n_atoms")
        self.v_min = self.safe_load(checkpoint, "v_min")
        self.v_max = self.safe_load(checkpoint, "v_max")
        self.n_actions = self.safe_load(checkpoint, "n_actions")
        self.n_steps = self.safe_load(checkpoint, "n_steps")
        self.omega = self.safe_load(checkpoint, "omega")
        self.omega_is = self.safe_load(checkpoint, "omega_is")
        self.replay_type = self.safe_load(checkpoint, "replay_type")
        self.loss_type = self.safe_load(checkpoint, "loss_type")
        self.network_type = self.safe_load(checkpoint, "network_type")
        self.training = self.safe_load(checkpoint, "training")
        self.epsilon_schedule = self.safe_load(checkpoint, "epsilon_schedule")
        self.current_step = self.safe_load(checkpoint, "current_step")
        self.epsilon = PiecewiseLinearSchedule(self.epsilon_schedule)

        # Update the loss function using the checkpoint.
        self.loss = self.get_loss(self.loss_type)

        # Update the agent's networks using the checkpoint.
        network = self.get_value_network(self.network_type)

        self.value_net = network()
        self.value_net.load_state_dict(self.safe_load(checkpoint, "value_net"))
        self.value_net.train(self.training)
        self.value_net.to(self.device)

        self.target_net = network()
        self.target_net.load_state_dict(self.safe_load(checkpoint, "target_net"))
        self.target_net.train(self.training)
        self.target_net.to(self.device)
        for param in self.target_net.parameters():
            param.requires_grad = False

        # Update the optimizer and replay buffer.
        replay_buffer = self.get_replay_buffer(self.replay_type, self.omega, self.omega_is, self.n_steps, self.gamma)
        self.buffer = replay_buffer(capacity=self.buffer_size, batch_size=self.batch_size)
        self.buffer.load(checkpoint_path, buffer_checkpoint_name)
        self.optimizer = optim.Adam(self.value_net.parameters(), lr=self.learning_rate, eps=self.adam_eps)

    def save(self, checkpoint_name, buffer_checkpoint_name=None):
        """
        Save the agent on the filesystem.
        :param checkpoint_name: the name of the checkpoint in which to save the agent
        :param buffer_checkpoint_name: the name of the checkpoint to save the replay buffer (None for default name)
        """

        # Create the agent checkpoint directory and file, if they do not exist.
        checkpoint_path = join(os.environ["CHECKPOINT_DIRECTORY"], checkpoint_name)
        FileSystem.create_directory_and_file(checkpoint_path)

        # Save the model.
        logging.info("Saving agent to the following file: " + checkpoint_path)
        torch.save({
            "gamma": self.gamma,
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "batch_size": self.batch_size,
            "target_update_interval": self.target_update_interval,
            "learning_starts": self.learning_starts,
            "kappa": self.kappa,
            "adam_eps": self.adam_eps,
            "n_atoms": self.n_atoms,
            "v_min": self.v_min,
            "v_max": self.v_max,
            "n_actions": self.n_actions,
            "n_steps": self.n_steps,
            "omega": self.omega,
            "omega_is": self.omega_is,
            "replay_type": self.replay_type,
            "loss_type": self.loss_type,
            "network_type": self.network_type,
            "training": self.training,
            "epsilon_schedule": self.epsilon_schedule,
            "current_step": self.current_step,
            "value_net": self.value_net.state_dict(),
            "target_net": self.target_net.state_dict(),
        }, checkpoint_path)

        # Save the replay buffer.
        self.buffer.save(checkpoint_path, buffer_checkpoint_name)
