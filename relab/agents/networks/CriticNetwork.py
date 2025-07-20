from typing import Optional

import torch
from torch import Tensor, nn

from relab import relab


class LinearCriticNetwork(nn.Module):
    """!
    @brief Class implementing a critic network based on continuous latent variables.
    """

    def __init__(
        self,
        n_actions: int = 18,
        n_continuous_vars: int = 10,
        n_latents: int = 512,
        n_layers: int = 4,
    ) -> None:
        """!
        Constructor.
        @param n_actions: the number of allowable actions
        @param n_continuous_vars: the number of continuous latent variables
        @param n_latents: the number of hidden neurons in the fully connected network
        @param n_layers: the number of layers in the fully connected network (at least one)
        """

        # Call the parent constructor.
        super().__init__()

        # @var net
        # Transition network that predicts the next state distribution.
        self.net = nn.Sequential()

        # Create the sequential network.
        for i in range(n_layers):

            # Add fully connected layers.
            if i == 0:
                self.net.append(nn.Linear(n_continuous_vars, n_latents))
            elif i != n_layers - 1:
                self.net.append(nn.Linear(n_latents, n_latents))
            else:
                self.net.append(nn.Linear(n_latents, n_actions))
                break

            # Add the activation function, if needed.
            self.net.append(nn.ReLU())

        # @var n_actions
        # Number of allowable actions in the environment.
        self.n_actions = n_actions

    def forward(self, states: Tensor) -> Tensor:
        """!
        Perform the forward pass through the network.
        @param states: the input states
        @return the Q-values as predicted by the critic
        """
        return self.net(states)


class ConvCriticNetwork(nn.Module):
    """!
    @brief Class implementing a critic network based on images.
    """

    def __init__(self, n_actions: int = 18, stack_size: Optional[int] = None) -> None:
        """!
        Constructor.
        @param n_actions: the number of actions available to the agent
        @param stack_size: the number of stacked frame in each observation, if None use the configuration
        """

        # Call the parent constructor.
        super().__init__()

        # @var stack_size
        # Number of stacked frames in each observation.
        self.stack_size = relab.config("stack_size", stack_size)

        # @var net
        # Complete network that processes images and outputs Q-values.
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
            nn.Linear(1024, n_actions),
        )

        # Initialize the weights.
        for name, param in self.named_parameters():
            if "weight" in name:
                torch.nn.init.kaiming_normal_(param, nonlinearity="leaky_relu")

    def forward(self, x: Tensor) -> Tensor:
        """!
        Perform the forward pass through the network.
        @param x: the observation
        @return the Q-values as predicted by the critic
        """
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=0)
        return self.net(x)
