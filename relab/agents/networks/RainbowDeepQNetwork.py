from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchrl.modules import NoisyLinear

import relab


class RainbowDeepQNetwork(nn.Module):
    """!
    @brief Implement the value network of rainbow DQN.
    """

    def __init__(
        self,
        n_actions: int = 18,
        n_atoms: int = 51,
        v_min: float = -10,
        v_max: float = 10,
        stack_size: Optional[int] = None,
    ) -> None:
        """!
        Constructor.
        @param n_actions: the number of actions available to the agent
        @param n_atoms: the number of atoms used to approximate the distribution over returns
        @param v_min: the minimum amount of returns
        @param v_max: the maximum amount of returns
        @param stack_size: the number of stacked frame in each observation, if None use the configuration
        """

        # Call the parent constructor.
        super().__init__()

        # @var device
        # Device on which the network is running.
        self.device = relab.device()

        # @var n_actions
        # Number of possible actions.
        self.n_actions = n_actions

        # @var n_atoms
        # Number of atoms used to approximate the distribution over returns.
        self.n_atoms = n_atoms

        # @var v_min
        # Minimum value of the support of the returns distribution.
        self.v_min = v_min

        # @var v_max
        # Maximum value of the support of the distribution over returns.
        self.v_max = v_max

        # @var delta_z
        # Step size between atoms in the support.
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

        # @var atoms
        # Support of the returns distribution.
        self.atoms = torch.tensor(
            [v_min + i * self.delta_z for i in range(self.n_atoms)]
        )
        self.atoms = self.atoms.unsqueeze(1).repeat(1, n_actions)
        self.atoms = self.atoms.to(relab.device())

        # @var stack_size
        # Number of stacked frames in each observation.
        self.stack_size = relab.config("stack_size", stack_size)

        # @var net
        # Complete network that processes images and outputs shared features.
        self.net = nn.Sequential(
            nn.Conv2d(self.stack_size, 32, 8, stride=4),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.LeakyReLU(0.01),
            nn.Flatten(start_dim=1),
            NoisyLinear(3136, 1024),
            nn.LeakyReLU(0.01),
            NoisyLinear(1024, 512),
            nn.LeakyReLU(0.01),
        )

        # @var value
        # Noisy linear layer that outputs the atoms of the state value
        # distribution.
        self.value = NoisyLinear(512, n_atoms)

        # @var advantage
        # Noisy linear layer that outputs the atoms of the action advantage
        # distributions.
        self.advantage = NoisyLinear(512, n_atoms * n_actions)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """!
        Perform the forward pass through the network.
        @param x: the observation
        @return a 3-tuple (returns, probabilities, log-probabilities)
        """

        # Ensure the input has the correct shape.
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=0)
        batch_size = x.shape[0]

        # Forward pass through the shared encoder.
        x = self.net(x)

        # Compute the Q-values.
        value = (
            self.value(x).view(batch_size, self.n_atoms, 1).repeat(1, 1, self.n_actions)
        )
        advantages = self.advantage(x).view(batch_size, self.n_atoms, self.n_actions)
        log_probs = (
            value
            + advantages
            - advantages.mean(dim=2).unsqueeze(dim=2).repeat(1, 1, self.n_actions)
        )
        probs = log_probs.softmax(dim=1)

        # Compute all atoms.
        atoms = self.atoms.unsqueeze(0).repeat(batch_size, 1, 1)

        # Return all atoms, their probabilities and log-probabilities.
        return atoms, probs, log_probs

    def q_values(self, x: Tensor) -> Tensor:
        """!
        Compute the Q-values for each action.
        @param x: the observation
        @return the Q-values
        """
        atoms, probs, _ = self(x)
        return (atoms * probs).sum(dim=1)


class RainbowImplicitQuantileNetwork(nn.Module):
    """!
    @brief Implement the value network of rainbow IQN.
    """

    def __init__(
        self, n_actions: int = 18, n_tau: int = 64, stack_size: Optional[int] = None
    ) -> None:
        """!
        Constructor.
        @param n_actions: the number of actions available to the agent
        @param n_tau: the size of the tau embedding
        @param stack_size: the number of stacked frame in each observation, if None use the configuration
        """

        # Call the parent constructor.
        super().__init__()

        # @var device
        # Device on which the network is running.
        self.device = relab.device()

        # @var n_actions
        # Number of possible actions.
        self.n_actions = n_actions

        # @var n_tau
        # Size of the tau embedding.
        self.n_tau = n_tau

        # @var conv_output
        # Cached output of the convolutional layers.
        self.conv_output = None

        # @var stack_size
        # Number of stacked frames in each observation.
        self.stack_size = relab.config("stack_size", stack_size)

        # @var conv_net
        # Convolutional network that processes the input images.
        self.conv_net = nn.Sequential(
            nn.Conv2d(self.stack_size, 32, 8, stride=4),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.LeakyReLU(0.01),
            nn.Flatten(start_dim=1),
        )

        # @var fc_net
        # Fully connected network that processes the combined features and tau
        # embeddings.
        self.fc_net = nn.Sequential(
            NoisyLinear(3136, 1024),
            nn.LeakyReLU(0.01),
            NoisyLinear(1024, 1024),
            nn.LeakyReLU(0.01),
        )

        # @var value
        # Noisy linear layer that outputs the state value.
        self.value = NoisyLinear(512, 1)

        # @var advantage
        # Noisy linear layer that outputs the action advantages.
        self.advantage = NoisyLinear(512, n_actions)

        # @var tau_fc1
        # Noisy linear layer that processes the tau embeddings.
        self.tau_fc1 = NoisyLinear(self.n_tau, 3136)

    def compute_conv_output(self, x: Tensor, invalidate_cache: bool = False) -> Tensor:
        """!
        Compute the output of the convolutional layers.
        @param x: the observation
        @param invalidate_cache: False if the cached output should be used, True to recompute the cached output
        @return the output of the convolutional layers
        """

        # Check if the cached convolutional output should be returned.
        if invalidate_cache is False:
            return self.conv_output

        # Compute forward pass through the convolutional layers.
        self.conv_output = self.conv_net(x)
        return self.conv_output

    def forward(
        self, x: Tensor, n_samples: int = 8, invalidate_cache: bool = True
    ) -> Tuple[Tensor, Tensor]:
        """!
        Perform the forward pass through the network.
        @param x: the observation
        @param n_samples: the number of taus to sample
        @param invalidate_cache: False if the cached output should be used, True to recompute the convolutional output
        @return a tuple (returns, sampled taus)
        """

        # Ensure the input has the correct shape.
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=0)
        batch_size = x.shape[0]

        # Compute the output of the convolutional layers.
        x = self.compute_conv_output(x, invalidate_cache=invalidate_cache)

        # Compute all the atoms.
        atoms = []
        taus = []
        for _ in range(n_samples):

            # Compute tau embeddings.
            tau = torch.rand([batch_size]).unsqueeze(dim=1).to(self.device)
            taus.append(tau)
            tau = torch.concat(
                [torch.cos(torch.pi * i * tau) for i in range(self.n_tau)], dim=1
            )
            tau = F.leaky_relu(self.tau_fc1(tau), 0.01)

            # Compute the Q-values.
            x_tau = self.fc_net(x * tau)
            value = self.value(x_tau).view(batch_size, 1).repeat(1, self.n_actions)
            advantages = self.advantage(x_tau).view(batch_size, self.n_actions)
            q_values = (
                value
                + advantages
                - advantages.mean(dim=1).unsqueeze(dim=1).repeat(1, self.n_actions)
            )
            atoms.append(q_values.unsqueeze(dim=1))

        # Concatenate all atoms and all taus along the atoms dimension.
        return torch.concat(atoms, dim=1), torch.concat(taus, dim=1)

    def q_values(
        self, x: Tensor, n_samples: int = 32, invalidate_cache: bool = True
    ) -> Tensor:
        """!
        Compute the Q-values for each action.
        @param x: the observation
        @param n_samples: the number of samples used to estimate the Q-values
        @param invalidate_cache: False if the cached output should be used, True to recompute the convolutional output
        @return the Q-values
        """
        atoms, _ = self(x, n_samples, invalidate_cache)
        return atoms.sum(dim=1) / n_samples
