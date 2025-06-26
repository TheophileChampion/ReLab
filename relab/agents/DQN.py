import logging
import math
from datetime import datetime
from enum import IntEnum
from functools import partial
from typing import Any, Callable, Dict, Optional

import numpy as np
import relab
import torch
from gymnasium import Env
from relab.agents.AgentInterface import AgentInterface, ReplayType
from relab.agents.networks.CategoricalDeepQNetworks import (
    CategoricalDeepQNetwork,
    NoisyCategoricalDeepQNetwork,
)
from relab.agents.networks.DeepQNetworks import DeepQNetwork, NoisyDeepQNetwork
from relab.agents.networks.DuelingDeepQNetworks import (
    DuelingDeepQNetwork,
    NoisyDuelingDeepQNetwork,
)
from relab.agents.networks.QuantileDeepQNetworks import (
    ImplicitQuantileNetwork,
    QuantileDeepQNetwork,
)
from relab.agents.networks.RainbowDeepQNetwork import (
    RainbowDeepQNetwork,
    RainbowImplicitQuantileNetwork,
)
from relab.agents.schedule.PiecewiseLinearSchedule import PiecewiseLinearSchedule
from relab.cpp.agents.memory import Experience
from relab.helpers.Serialization import get_adam_optimizer, safe_load_state_dict
from relab.helpers.Typing import (
    ActionType,
    AttributeNames,
    Checkpoint,
    Config,
    Loss,
    ObservationType,
)
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss, HuberLoss, MSELoss, SmoothL1Loss


class LossType(IntEnum):
    """!
    The type of loss functions supported by the DQN agents.
    """

    # @var DQN_MSE
    # Q-learning loss using Mean Square Error.
    DQN_MSE = 0

    # @var DQN_SL1
    # Q-learning loss using Smooth L1 loss.
    DQN_SL1 = 1

    # @var DDQN_MSE
    # Double Q-learning loss using Mean Square Error.
    DDQN_MSE = 2

    # @var DDQN_SL1
    # Double Q-learning loss using Smooth L1 loss.
    DDQN_SL1 = 3

    # @var KL_DIVERGENCE
    # KL-divergence loss for Categorical DQN.
    KL_DIVERGENCE = 4

    # @var QUANTILE
    # Huber quantile regression loss for QR-DQN.
    QUANTILE = 5

    # @var IMPLICIT_QUANTILE
    # Loss function for Implicit Quantile Networks (IQN).
    IMPLICIT_QUANTILE = 6

    # @var RAINBOW
    # Combined loss function for Rainbow DQN.
    RAINBOW = 7

    # @var RAINBOW_IQN
    # Combined loss function for Rainbow with IQN.
    RAINBOW_IQN = 8


class NetworkType(IntEnum):
    """!
    The type of networks supported by the DQN agents.
    """

    # @var DEFAULT
    # Standard Deep Q-Network architecture.
    DEFAULT = 0

    # @var NOISY
    # DQN with noisy linear layers for exploration.
    NOISY = 1

    # @var DUELING
    # Dueling architecture separating state value and action advantage.
    DUELING = 2

    # @var NOISY_DUELING
    # Dueling architecture with noisy linear layers.
    NOISY_DUELING = 3

    # @var CATEGORICAL
    # Categorical Deep Q-network architecture.
    CATEGORICAL = 4

    # @var NOISY_CATEGORICAL
    # Categorical Deep Q-network network with noisy linear layers.
    NOISY_CATEGORICAL = 5

    # @var QUANTILE
    # Network for Quantile Regression DQN.
    QUANTILE = 6

    # @var IMPLICIT_QUANTILE
    # Network architecture for Implicit Quantile Networks.
    IMPLICIT_QUANTILE = 7

    # @var RAINBOW
    # Combined architecture used in Rainbow DQN.
    RAINBOW = 8

    # @var RAINBOW_IQN
    # Rainbow architecture with Implicit Quantile Networks.
    RAINBOW_IQN = 9


class DQN(AgentInterface):
    """!
    @brief Implements a Deep Q-Network.

    @details
    This implementation is based on the paper:

    <b>Human-level control through deep reinforcement learning</b>,
    published in Nature, 2015.

    Authors:
    - Volodymyr Mnih
    - Koray Kavukcuoglu
    - David Silver
    - Andrei A. Rusu
    - Joel Veness
    - Marc G. Bellemare
    - Alex Graves
    - Martin Riedmiller
    - Andreas K. Fidjeland
    - Georg Ostrovski, et al.

    The paper introduced the DQN algorithm, combining Q-learning with deep neural
    networks to achieve human-level performance in Atari 2600 games.
    """

    def __init__(
        self,
        gamma: float = 0.99,
        learning_rate: float = 0.00001,
        buffer_size: int = 1000000,
        batch_size: int = 32,
        learning_starts: int = 200000,
        kappa: Optional[float] = None,
        target_update_interval: int = 40000,
        adam_eps: float = 1.5e-4,
        n_actions: int = 18,
        n_atoms: Optional[int] = None,
        v_min: Optional[float] = None,
        v_max: Optional[float] = None,
        n_steps: int = 1,
        training: bool = True,
        replay_type: ReplayType = ReplayType.DEFAULT,
        loss_type: LossType = LossType.DQN_SL1,
        network_type: NetworkType = NetworkType.DEFAULT,
        omega: float = 1.0,
        omega_is: float = 1.0,
        epsilon_schedule: Any = None,
    ) -> None:
        """!
        Create a DQN agent.
        @param gamma: the discount factor
        @param learning_rate: the learning rate
        @param buffer_size: the size of the replay buffer
        @param batch_size: the size of the batches sampled from the replay buffer
        @param learning_starts: the step at which learning starts
        @param kappa: the kappa parameter of the quantile Huber loss see Equation (10) in QR-DQN paper
        @param target_update_interval: number of training steps between two synchronization of the target
        @param adam_eps: the epsilon parameter of the Adam optimizer
        @param n_actions: the number of actions available to the agent
        @param n_atoms: the number of atoms used to approximate the distribution over returns
        @param v_min: the minimum amount of returns (only used for distributional DQN)
        @param v_max: the maximum amount of returns (only used for distributional DQN)
        @param training: True if the agent is being trained, False otherwise
        @param n_steps: the number of steps for which rewards are accumulated in multistep Q-learning
        @param omega: the prioritization exponent
        @param omega_is: the important sampling exponent
        @param replay_type: the type of replay buffer
        @param loss_type: the loss to use during gradient descent
        @param network_type: the network architecture to use for the value and target networks
        @param epsilon_schedule: the schedule for the exploration parameter epsilon as a list of tuple, i.e.,
            [(step_1, value_1), (step_2, value_2), ..., (step_n, value_n)]
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
        # Learning rate for the optimizer.
        self.learning_rate = learning_rate

        # @var buffer_size
        # Maximum number of transitions stored in the replay buffer.
        self.buffer_size = buffer_size

        # @var batch_size
        # Number of transitions sampled per learning update.
        self.batch_size = batch_size

        # @var target_update_interval
        # Number of training steps between target network updates.
        self.target_update_rate = target_update_interval

        # @var learning_starts
        # Step count at which learning begins.
        self.learning_starts = learning_starts

        # @var kappa
        # Parameter for the quantile Huber loss (used in QR-DQN).
        self.kappa = kappa

        # @var adam_eps
        # Epsilon parameter for the Adam optimizer.
        self.adam_eps = adam_eps

        # @var n_atoms
        # Number of atoms used to approximate the return distribution.
        self.n_atoms = n_atoms

        # @var v_min
        # Minimum value for the return distribution support.
        self.v_min = v_min

        # @var v_max
        # Maximum value for the return distribution support.
        self.v_max = v_max

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

        # @var loss_type
        # Type of loss function used for training.
        self.loss_type = loss_type

        # @var network_type
        # Type of neural network architecture used.
        self.network_type = network_type

        # @var epsilon_schedule
        # Schedule for the exploration parameter epsilon.
        self.epsilon_schedule = (
            [(0, 1), (self.learning_starts, 1), (1e6, 0.1), (10e6, 0.01)]
            if epsilon_schedule is None
            else epsilon_schedule
        )

        # @var epsilon
        # Scheduler for the exploration parameter epsilon.
        self.epsilon = PiecewiseLinearSchedule(self.epsilon_schedule)

        # @var loss
        # Loss function used for computing gradients.
        self.loss = self.get_loss(self.loss_type)

        # @var value_net
        # The value network that approximates the Q-value function.
        self.value_net = self.get_value_network(self.network_type)

        # @var target_net
        # The target network, which is a copy of the value network synchronized
        # periodically.
        self.target_net = self.get_value_network(self.network_type)
        self.update_target_network()
        for param in self.target_net.parameters():
            param.requires_grad = False

        # @var optimizer
        # Adam optimizer for training the value network.
        self.optimizer = get_adam_optimizer(
            [self.value_net],
            self.learning_rate,
            self.adam_eps,
        )

    def get_loss(self, loss_type: LossType) -> Callable:
        """!
        Retrieve the loss requested as parameters.
        @param loss_type: the loss to use during gradient descent
        @return the loss
        """
        # @cond IGNORED_BY_DOXYGEN
        return {
            LossType.KL_DIVERGENCE: self.categorical_kl_divergence,
            LossType.QUANTILE: partial(self.quantile_loss, kappa=self.kappa),
            LossType.RAINBOW: self.rainbow_loss,
            LossType.RAINBOW_IQN: partial(self.rainbow_iqn_loss, kappa=self.kappa),
            LossType.IMPLICIT_QUANTILE: partial(
                self.implicit_quantile_loss, kappa=self.kappa
            ),
            LossType.DQN_MSE: partial(
                self.q_learning_loss, loss_fc=MSELoss(reduction="none")
            ),
            LossType.DQN_SL1: partial(
                self.q_learning_loss, loss_fc=SmoothL1Loss(reduction="none")
            ),
            LossType.DDQN_MSE: partial(
                self.q_learning_loss, loss_fc=MSELoss(reduction="none"), double_ql=True
            ),
            LossType.DDQN_SL1: partial(
                self.q_learning_loss,
                loss_fc=SmoothL1Loss(reduction="none"),
                double_ql=True,
            ),
        }[loss_type]
        # @endcond

    def get_value_network(self, network_type: NetworkType) -> nn.Module:
        """!
        Retrieve the constructor of the value network requested as parameters.
        @param network_type: the network architecture to use for the value and target networks
        @return the constructor of the value network
        """
        # @cond IGNORED_BY_DOXYGEN
        network = {
            NetworkType.DEFAULT: partial(DeepQNetwork, self.n_actions),
            NetworkType.NOISY: partial(NoisyDeepQNetwork, self.n_actions),
            NetworkType.DUELING: partial(DuelingDeepQNetwork, self.n_actions),
            NetworkType.NOISY_DUELING: partial(
                NoisyDuelingDeepQNetwork, self.n_actions
            ),
            NetworkType.QUANTILE: partial(
                QuantileDeepQNetwork, self.n_actions, self.n_atoms
            ),
            NetworkType.IMPLICIT_QUANTILE: partial(
                ImplicitQuantileNetwork, self.n_actions
            ),
            NetworkType.RAINBOW_IQN: partial(
                RainbowImplicitQuantileNetwork, self.n_actions
            ),
            NetworkType.CATEGORICAL: partial(
                CategoricalDeepQNetwork,
                self.n_actions,
                self.n_atoms,
                self.v_min,
                self.v_max,
            ),
            NetworkType.NOISY_CATEGORICAL: partial(
                NoisyCategoricalDeepQNetwork,
                self.n_actions,
                self.n_atoms,
                self.v_min,
                self.v_max,
            ),
            NetworkType.RAINBOW: partial(
                RainbowDeepQNetwork,
                self.n_actions,
                self.n_atoms,
                self.v_min,
                self.v_max,
            ),
        }[network_type]()
        network.train(self.training)
        network.to(self.device)
        return network
        # @endcond

    def update_target_network(self) -> None:
        """!
        Synchronize the target with the value network.
        """
        self.target_net.load_state_dict(self.value_net.state_dict())

    def step(self, obs: ObservationType) -> ActionType:
        """!
        Select the next action to perform in the environment.
        @param obs: the observation available to make the decision
        @return the next action to perform
        """
        # @cond IGNORED_BY_DOXYGEN
        if not self.training or np.random.random() > self.epsilon(self.current_step):
            return torch.argmax(self.value_net.q_values(obs), dim=1).item()
        return np.random.choice(self.n_actions)
        # @endcond

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
        # @endcond

    def learn(self) -> Optional[Dict[str, Any]]:
        """!
        Perform one step of gradient descent on the value network.
        """

        # Synchronize the target with the value network (if needed).
        if self.current_step % self.target_update_rate == 0:
            self.update_target_network()

        # Sample the replay buffer.
        obs, actions, rewards, done, next_obs = self.buffer.sample()

        # Compute the Q-value loss.
        loss = self.loss(obs, actions, rewards, done, next_obs)

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

    def q_learning_loss(
        self,
        obs: Tensor,
        actions: Tensor,
        rewards: Tensor,
        done: Tensor,
        next_obs: Tensor,
        loss_fc: Loss,
        double_ql: bool = False,
    ) -> Tensor:
        """!
        Compute the loss of the standard or double Q-learning algorithm.
        @param obs: the observations at time t
        @param actions: the actions at time t
        @param rewards: the reward obtained when taking the actions while seeing the observations at time t
        @param done: whether the episodes ended
        @param next_obs: the observation at time t + 1
        @param loss_fc: the loss function to use to compare target and prediction
        @param double_ql: False for standard Q-learning, True for Double Q-learning
        @return the Q-value loss
        """

        next_values = self.target_net.q_values(next_obs)
        if double_ql is True:
            # Chose the best actions according to the value network,
            # and evaluate them using the target network.
            next_actions = torch.argmax(self.value_net.q_values(next_obs), dim=1)
            next_actions = next_actions.detach().squeeze()
            next_values = next_values[range(self.batch_size), next_actions]
        else:
            # Chose and evaluate the best actions using the target network.
            next_values = torch.max(next_values, dim=1).values
        next_values = next_values.detach()

        # Compute the Q-value loss.
        mask = torch.logical_not(done).float()
        y = rewards + mask * math.pow(self.gamma, self.n_steps) * next_values
        x = self.value_net.q_values(obs)
        loss = loss_fc(x[range(self.batch_size), actions.squeeze()], y)
        return loss

    def categorical_kl_divergence(
        self,
        obs: Tensor,
        actions: Tensor,
        rewards: Tensor,
        done: Tensor,
        next_obs: Tensor,
    ) -> Tensor:
        """!
        Compute the loss of the categorical algorithm.
        @param obs: the observations at time t
        @param actions: the actions at time t
        @param rewards: the reward obtained when taking the actions while seeing the observations at time t
        @param done: whether the episodes ended
        @param next_obs: the observation at time t + 1
        @return the categorical loss
        """

        # Compute the best actions at time t + 1.
        next_atoms, next_probs, _ = self.target_net(next_obs)
        next_q_values = (next_atoms * next_probs).sum(dim=1)
        next_actions = torch.argmax(next_q_values, dim=1).squeeze()

        # Retrieve the atoms and probabilities corresponding to the best
        # actions at time t + 1.
        batch_ids = range(self.batch_size)
        next_atoms = next_atoms[batch_ids, :, next_actions]
        next_probs = next_probs[batch_ids, :, next_actions]

        # Compute the new atom positions using the Bellman update.
        next_atoms = (
            rewards.unsqueeze(dim=1).repeat(1, self.n_atoms)
            + math.pow(self.gamma, self.n_steps) * next_atoms
        )
        next_atoms = torch.clamp(next_atoms, self.v_min, self.v_max)

        # Compute the projected distribution over returns.
        target_probs = torch.zeros_like(next_probs)
        for j in range(self.n_atoms):
            atom = (next_atoms[:, j] - self.v_min) / self.target_net.delta_z
            lower = torch.floor(atom).int()
            upper = torch.ceil(atom).int()
            target_probs[batch_ids, lower] += next_probs[batch_ids, j] * (upper - atom)
            mask = torch.logical_not(torch.eq(lower, upper))
            target_probs[batch_ids, upper] += (
                mask * next_probs[batch_ids, j] * (atom - lower)
            )

        # Compute the predicted return log-probabilities.
        _, _, log_probs = self.value_net(obs)
        log_probs = log_probs[batch_ids, :, actions.squeeze()]

        # Compute the categorical loss.
        loss_fc = CrossEntropyLoss(reduction="none")
        loss = loss_fc(log_probs, target_probs.detach())
        return loss

    def rainbow_loss(
        self,
        obs: Tensor,
        actions: Tensor,
        rewards: Tensor,
        done: Tensor,
        next_obs: Tensor,
    ) -> Tensor:
        """!
        Compute the loss of the rainbow DQN.
        @param obs: the observations at time t
        @param actions: the actions at time t
        @param rewards: the reward obtained when taking the actions while seeing the observations at time t
        @param done: whether the episodes ended
        @param next_obs: the observation at time t + 1
        @return the rainbow loss
        """

        # Compute the best actions at time t + 1 using the value network.
        next_actions = (
            torch.argmax(self.value_net.q_values(next_obs), dim=1).detach().squeeze()
        )

        # Retrieve the atoms and probabilities corresponding to the best
        # actions at time t + 1.
        batch_ids = range(self.batch_size)
        next_atoms, next_probs, _ = self.target_net(next_obs)
        next_atoms = next_atoms[batch_ids, :, next_actions]
        next_probs = next_probs[batch_ids, :, next_actions]

        # Compute the new atom positions using the Bellman update.
        next_atoms = (
            rewards.unsqueeze(dim=1).repeat(1, self.n_atoms)
            + math.pow(self.gamma, self.n_steps) * next_atoms
        )
        next_atoms = torch.clamp(next_atoms, self.v_min, self.v_max)

        # Compute the projected distribution over returns.
        target_probs = torch.zeros_like(next_probs)
        for j in range(self.n_atoms):
            atom = (next_atoms[:, j] - self.v_min) / self.target_net.delta_z
            lower = torch.floor(atom).int()
            upper = torch.ceil(atom).int()
            target_probs[batch_ids, lower] += next_probs[batch_ids, j] * (upper - atom)
            mask = torch.logical_not(torch.eq(lower, upper))
            target_probs[batch_ids, upper] += (
                mask * next_probs[batch_ids, j] * (atom - lower)
            )

        # Compute the predicted return log-probabilities.
        _, _, log_probs = self.value_net(obs)
        log_probs = log_probs[range(self.batch_size), :, actions.squeeze()]

        # Compute the categorical loss.
        loss_fc = CrossEntropyLoss(reduction="none")
        loss = loss_fc(log_probs, target_probs.detach())
        return loss

    def quantile_loss(
        self,
        obs: Tensor,
        actions: Tensor,
        rewards: Tensor,
        done: Tensor,
        next_obs: Tensor,
        kappa: float = 1.0,
    ) -> Tensor:
        """!
        Compute the loss of the quantile regression algorithm.
        @param obs: the observations at time t
        @param actions: the actions at time t
        @param rewards: the reward obtained when taking the actions while seeing the observations at time t
        @param done: whether the episodes ended
        @param next_obs: the observation at time t + 1
        @param kappa: the kappa parameter of the quantile Huber loss see Equation (10) in QR-DQN paper
        @return the categorical loss
        """
        # @cond IGNORED_BY_DOXYGEN

        # Compute the best actions at time t + 1.
        next_atoms = self.target_net(next_obs)
        next_q_values = next_atoms.sum(dim=1) / self.n_atoms
        next_actions = torch.argmax(next_q_values, dim=1)

        # Compute the new atom positions using the Bellman update.
        batch_ids = range(self.batch_size)
        next_atoms = next_atoms[batch_ids, :, next_actions.squeeze()]
        next_atoms = (
            rewards.unsqueeze(dim=1).repeat(1, self.n_atoms)
            + math.pow(self.gamma, self.n_steps) * next_atoms
        )

        # Compute the predicted atoms (canonical return).
        atoms = self.value_net(obs)
        atoms = atoms[batch_ids, :, actions.squeeze()]

        # Compute the quantile Huber loss.
        huber_loss = HuberLoss(reduction="none", delta=kappa)
        loss = torch.zeros([self.batch_size]).to(self.device)
        for i in range(self.n_atoms):
            tau = (i + 0.5) / self.n_atoms
            for j in range(self.n_atoms):
                next_atom_j = next_atoms[:, j]
                atom_i = atoms[:, i]
                mask = torch.where(next_atom_j - atom_i < 0, 1.0, 0.0)
                loss += (
                    torch.abs(tau - mask).to(self.device)
                    * huber_loss(next_atom_j, atom_i)
                    / kappa
                )
        loss /= self.n_atoms
        return loss
        # @endcond

    def implicit_quantile_loss(
        self,
        obs: Tensor,
        actions: Tensor,
        rewards: Tensor,
        done: Tensor,
        next_obs: Tensor,
        kappa: float = 1.0,
    ) -> Tensor:
        """!
        Compute the loss of the quantile regression algorithm.
        @param obs: the observations at time t
        @param actions: the actions at time t
        @param rewards: the reward obtained when taking the actions while seeing the observations at time t
        @param done: whether the episodes ended
        @param next_obs: the observation at time t + 1
        @param kappa: the kappa parameter of the quantile Huber loss see Equation (10) in QR-DQN paper
        @return the categorical loss
        """
        # @cond IGNORED_BY_DOXYGEN

        # Compute the best actions at time t + 1.
        next_q_values = self.target_net.q_values(next_obs)
        next_actions = torch.argmax(next_q_values, dim=1)

        # Compute the new atom positions using the Bellman update.
        batch_ids = range(self.batch_size)
        next_atoms, _ = self.target_net(next_obs, self.n_atoms, False)
        next_atoms = next_atoms[batch_ids, :, next_actions.squeeze()]
        next_atoms = (
            rewards.unsqueeze(dim=1).repeat(1, self.n_atoms)
            + math.pow(self.gamma, self.n_steps) * next_atoms
        )

        # Compute the predicted atoms (canonical return).
        atoms, taus = self.value_net(obs, n_samples=self.n_atoms)
        atoms = atoms[batch_ids, :, actions.squeeze()]

        # Compute the quantile Huber loss.
        huber_loss = HuberLoss(reduction="none", delta=kappa)
        loss = torch.zeros([self.batch_size]).to(self.device)
        for i in range(self.n_atoms):
            atom_i = atoms[:, i]
            for j in range(self.n_atoms):
                next_atom_j = next_atoms[:, j]
                mask = torch.where(next_atom_j - atom_i < 0, 1.0, 0.0)
                loss += (
                    torch.abs(taus[:, i] - mask).to(self.device)
                    * huber_loss(next_atom_j, atom_i)
                    / kappa
                )
        loss /= self.n_atoms
        return loss
        # @endcond

    def rainbow_iqn_loss(
        self,
        obs: Tensor,
        actions: Tensor,
        rewards: Tensor,
        done: Tensor,
        next_obs: Tensor,
        kappa: float = 1.0,
    ) -> Tensor:
        """!
        Compute the loss of the rainbow IQN.
        @param obs: the observations at time t
        @param actions: the actions at time t
        @param rewards: the reward obtained when taking the actions while seeing the observations at time t
        @param done: whether the episodes ended
        @param next_obs: the observation at time t + 1
        @param kappa: the kappa parameter of the quantile Huber loss see Equation (3) in IQN paper
        @return the rainbow IQN loss
        """
        # @cond IGNORED_BY_DOXYGEN

        # Compute the best actions at time t + 1 using the value network.
        next_actions = torch.argmax(self.value_net.q_values(next_obs), dim=1).detach()

        # Compute the new atom positions using the Bellman update.
        batch_ids = range(self.batch_size)
        next_atoms, _ = self.target_net(next_obs, n_samples=self.n_atoms)
        next_atoms = next_atoms[batch_ids, :, next_actions.squeeze()]
        next_atoms = (
            rewards.unsqueeze(dim=1).repeat(1, self.n_atoms)
            + math.pow(self.gamma, self.n_steps) * next_atoms
        )

        # Compute the predicted atoms (canonical return) at time t.
        atoms, taus = self.value_net(obs, n_samples=self.n_atoms)
        atoms = atoms[batch_ids, :, actions.squeeze()]

        # Compute the quantile Huber loss.
        huber_loss = HuberLoss(reduction="none", delta=kappa)
        loss = torch.zeros([self.batch_size]).to(self.device)
        for i in range(self.n_atoms):
            atom_i = atoms[:, i]
            for j in range(self.n_atoms):
                next_atom_j = next_atoms[:, j]
                mask = torch.where(next_atom_j - atom_i < 0, 1.0, 0.0)
                loss += (
                    torch.abs(taus[:, i] - mask).to(self.device)
                    * huber_loss(next_atom_j, atom_i)
                    / kappa
                )
        loss /= self.n_atoms
        return loss
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

            # Load the epsilon scheduler and update the loss function using the checkpoint.
            self.epsilon = PiecewiseLinearSchedule(self.epsilon_schedule)
            self.loss = self.get_loss(self.loss_type)

            # Update the agent's networks using the checkpoint.
            self.value_net = self.get_value_network(self.network_type)
            safe_load_state_dict(self.value_net, checkpoint, "value_net")

            self.target_net = self.get_value_network(self.network_type)
            safe_load_state_dict(self.target_net, checkpoint, "target_net")
            for param in self.target_net.parameters():
                param.requires_grad = False

            # Update the optimizer.
            self.optimizer = get_adam_optimizer(
                [self.value_net], self.learning_rate, self.adam_eps, checkpoint
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
            "buffer_size": self.buffer_size,
            "batch_size": self.batch_size,
            "target_update_rate": self.target_update_rate,
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
            "epsilon_schedule": self.epsilon_schedule,
            "value_net": self.value_net.state_dict(),
            "target_net": self.target_net.state_dict(),
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
