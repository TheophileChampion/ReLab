from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import relab


class QuantileDeepQNetwork(nn.Module):
    """!
    @brief Implement the value network of QR-DQN.
    """

    def __init__(
        self, n_atoms: int = 21, n_actions: int = 18, stack_size: Optional[int] = None
    ) -> None:
        """!
        Constructor.
        @param n_atoms: the number of atoms used to approximate the distribution over returns
        @param n_actions: the number of actions available to the agent
        @param stack_size: the number of stacked frame in each observation, if None use the configuration
        """

        # Call the parent constructor.
        super().__init__()

        # @var n_atoms
        # Number of atoms used to approximate the distribution over returns.
        self.n_atoms = n_atoms

        # @var n_actions
        # Number of possible actions.
        self.n_actions = n_actions

        # @var stack_size
        # Number of stacked frames in each observation.
        self.stack_size = relab.config("stack_size", stack_size)

        # @var net
        # Complete network that processes images and outputs quantile values.
        self.net = nn.Sequential(
            nn.Conv2d(self.stack_size, 32, 8, stride=4),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.LeakyReLU(0.01),
            nn.Flatten(start_dim=1),
            nn.Linear(3136, 1024),
            nn.LeakyReLU(0.01),
            nn.Linear(1024, n_atoms * n_actions),
        )

        # Initialize the weights.
        for name, param in self.named_parameters():
            if "weight" in name:
                torch.nn.init.kaiming_normal_(param, nonlinearity="leaky_relu")

    def forward(self, x: Tensor) -> Tensor:
        """!
        Perform the forward pass through the network.
        @param x: the observation
        @return the returns
        """

        # Ensure the input has the correct shape.
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=0)
        batch_size = x.shape[0]

        # Compute forward pass.
        return self.net(x).view(batch_size, self.n_atoms, self.n_actions)

    def q_values(self, x: Tensor) -> Tensor:
        """!
        Compute the Q-values for each action.
        @param x: the observation
        @return the Q-values
        """
        return self(x).sum(dim=1) / self.n_atoms


class ImplicitQuantileNetwork(nn.Module):
    """!
    @brief Implement the value network of the IQN.
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
            nn.Linear(3136, 1024), nn.LeakyReLU(0.01), nn.Linear(1024, n_actions)
        )

        # @var tau_fc1
        # Linear layer that embeds the sampled tau values.
        self.tau_fc1 = nn.Linear(self.n_tau, 3136)

        # Initialize the weights.
        for name, param in self.named_parameters():
            if "weight" in name:
                torch.nn.init.kaiming_normal_(param, nonlinearity="leaky_relu")

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
        self.conv_output = self.conv_net(x).view(x.shape[0], -1)
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

            # Compute the output
            atoms_tau = self.fc_net(x * tau)
            atoms.append(atoms_tau.view(batch_size, 1, self.n_actions))

        # Concatenate all atoms and taus along the atoms dimension.
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
