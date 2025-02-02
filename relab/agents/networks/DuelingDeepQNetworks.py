from typing import Optional, Tuple, Union

import relab
import torch
from torch import Tensor, nn
from torchrl.modules import NoisyLinear


class DuelingDeepQNetwork(nn.Module):
    """!
    @brief Implement the value network of a dueling DQN.
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

        # @var conv_net
        # Convolutional network that processes the input images.
        self.conv_net = nn.Sequential(
            nn.Conv2d(self.stack_size, 32, 8, stride=4),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.LeakyReLU(0.01),
        )

        # @var fc_net
        # Fully connected network that processes flattened convolutional
        # features.
        self.fc_net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(3136, 1024),
            nn.LeakyReLU(0.01),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.01),
        )

        # @var value
        # Linear layer that outputs the state value.
        self.value = nn.Linear(512, 1)

        # @var advantage
        # Linear layer that outputs the action advantages.
        self.advantage = nn.Linear(512, n_actions)

        # Initialize the weights.
        for name, param in self.named_parameters():
            if "weight" in name:
                torch.nn.init.kaiming_normal_(param, nonlinearity="leaky_relu")

    def forward(self, x: Tensor, return_all: bool = False) -> Tensor:
        """!
        Perform the forward pass through the network.
        @param x: the observations
        @param return_all: True, if the Q-value, state value and advantage must be returned, False for only the Q-value
        @return a 3-tuple (Q-values, state values, advantages) or simply the Q-values
        """

        # Forward pass through the shared encoder.
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=0)
        x = self.conv_net(x)
        x = self.fc_net(x)

        # Compute the Q-values.
        value = self.value(x)
        advantages = self.advantage(x)
        q_values = value + advantages - advantages.mean()
        return (q_values, value, advantages) if return_all is True else q_values

    def q_values(self, x: Tensor) -> Tensor:
        """!
        Compute the Q-values for each action.
        @param x: the observation
        @return the Q-values
        """
        return self(x)


class NoisyDuelingDeepQNetwork(nn.Module):
    """!
    @brief Implement the value network of a dueling DQN with noisy linear layers.
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

        # @var conv_net
        # Convolutional network that processes the input images.
        self.conv_net = nn.Sequential(
            nn.Conv2d(self.stack_size, 32, 8, stride=4),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.LeakyReLU(0.01),
        )

        # @var fc_net
        # Fully connected network that processes flattened convolutional
        # features.
        self.fc_net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(3136, 1024),
            nn.LeakyReLU(0.01),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.01),
        )

        # @var value
        # Noisy linear layer that outputs the state value.
        self.value = NoisyLinear(512, 1)

        # @var advantage
        # Noisy linear layer that outputs the action advantages.
        self.advantage = NoisyLinear(512, n_actions)

        # Initialize the weights.
        for name, param in self.named_parameters():
            if "weight" in name:
                torch.nn.init.kaiming_normal_(param, nonlinearity="leaky_relu")

    def forward(
        self, x: Tensor, return_all: bool = False
    ) -> Union[Tuple[Tensor, Tensor, Tensor], Tensor]:
        """!
        Perform the forward pass through the network.
        @param x: the observations
        @param return_all: True, if the Q-value, state value and advantage must be returned, False for only the Q-value
        @return a tuple containing the state value and advantages
        @return a 3-tuple (Q-values, state values, advantages) or simply the Q-values
        """

        # Forward pass through the shared encoder.
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=0)
        x = self.conv_net(x)
        x = self.fc_net(x)

        # Compute the Q-values.
        value = self.value(x)
        advantages = self.advantage(x)
        q_values = (
            value
            + advantages
            - advantages.mean(dim=1).unsqueeze(dim=1).repeat(1, self.n_actions)
        )
        return (q_values, value, advantages) if return_all is True else q_values

    def q_values(self, x: Tensor) -> Tensor:
        """!
        Compute the Q-values for each action.
        @param x: the observation
        @return the Q-values
        """
        return self(x)
