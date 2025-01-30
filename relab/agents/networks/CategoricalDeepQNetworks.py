from typing import Tuple, Optional

from torch import nn, Tensor
import torch

import relab


class CategoricalDeepQNetwork(nn.Module):
    """!
    @brief Implements a value network for the categorical DQN.
    """

    def __init__(
        self,
        n_actions: int = 18,
        n_atoms: int = 21,
        v_min: float = -10,
        v_max: float = 10,
        stack_size: Optional[int] = None
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

        # @var n_atoms
        # Number of atoms used to approximate the distribution over returns.
        self.n_atoms = n_atoms

        # @var n_actions
        # Number of possible actions.
        self.n_actions = n_actions

        # @var v_min
        # Minimum value of the support of the distribution over returns.
        self.v_min = v_min

        # @var v_max
        # Maximum value of the support of the distribution over returns.
        self.v_max = v_max

        # @var delta_z
        # Step size between atoms in the support.
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

        # @var atoms
        # Support of the returns distribution.
        self.atoms = torch.arange(v_min, v_max + 1, self.delta_z)
        self.atoms = self.atoms.unsqueeze(1).repeat(1, n_actions)
        self.atoms = self.atoms.to(relab.device())

        # @var stack_size
        # Number of stacked frames in each observation.
        self.stack_size = \
            relab.config("stack_size") if stack_size is None else stack_size

        # @var net
        # Complete network that processes images and outputs atom logits.
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
            nn.Linear(1024, n_atoms * n_actions)
        )

        # Initialize the weights.
        for name, param in self.named_parameters():
            if "weight" in name:
                torch.nn.init.kaiming_normal_(param, nonlinearity="leaky_relu")

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

        # Compute forward pass.
        log_probs = self.net(x).view(batch_size, self.n_atoms, self.n_actions)
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


class NoisyCategoricalDeepQNetwork(nn.Module):
    """!
    @brief Implements a noisy value network for the categorical DQN.
    """

    def __init__(
        self,
        n_actions: int = 18,
        n_atoms: int = 21,
        v_min: float = -10,
        v_max: float = 10,
        stack_size: Optional[int] = None
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

        # @var n_atoms
        # Number of atoms used to approximate the distribution over returns.
        self.n_atoms = n_atoms

        # @var n_actions
        # Number of possible actions.
        self.n_actions = n_actions

        # @var v_min
        # Minimum value of the support of the returns distribution.
        self.v_min = v_min

        # @var v_max
        # Maximum value of the support of the return distribution.
        self.v_max = v_max

        # @var delta_z
        # Step size between atoms in the support.
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

        # @var atoms
        # Support of the returns distribution.
        self.atoms = torch.arange(v_min, v_max + 1, self.delta_z)
        self.atoms = self.atoms.unsqueeze(1).repeat(1, n_actions)
        self.atoms = self.atoms.to(relab.device())

        # @var stack_size
        # Number of stacked frames in each observation.
        self.stack_size = relab.config(
            "stack_size") if stack_size is None else stack_size

        # @var net
        # Complete network that processes images and outputs atom logits.
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
            nn.Linear(1024, n_atoms * n_actions)
        )

        # Initialize the weights.
        for name, param in self.named_parameters():
            if "weight" in name:
                torch.nn.init.kaiming_normal_(param, nonlinearity="leaky_relu")

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

        # Compute forward pass.
        log_probs = self.net(x).view(batch_size, self.n_atoms, self.n_actions)
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
