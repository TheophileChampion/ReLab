from typing import Optional

from torch import nn, Tensor
import torch

from relab import relab


class DeepQNetwork(nn.Module):
    """!
    Implement the DQN's value network from:
    Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves,
    Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al.
    Human-level control through deep reinforcement learning. nature, 2015.
    """

    def __init__(self, n_actions : int = 18, stack_size : Optional[int] = None) -> None:
        """!
        Constructor.
        @param n_actions: the number of actions available to the agent
        @param stack_size: the number of stacked frame in each observation, if None use the configuration
        """

        # Call the parent constructor.
        super().__init__()

        ## @var stack_size
        # Number of stacked frames in each observation.
        self.stack_size = relab.config("stack_size") if stack_size is None else stack_size

        ## @var net
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
            nn.Linear(1024, n_actions)
        )

        # Initialize the weights.
        for name, param in self.named_parameters():
            if "weight" in name:
                torch.nn.init.kaiming_normal_(param, nonlinearity="leaky_relu")

    def forward(self, x : Tensor) -> Tensor:
        """!
        Perform the forward pass through the network.
        @param x: the observation
        @return the Q-values
        """
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=0)
        return self.net(x)

    def q_values(self, x : Tensor) -> Tensor:
        """!
        Compute the Q-values for each action.
        @param x: the observation
        @return the Q-values
        """
        return self(x)


class NoisyDeepQNetwork(nn.Module):
    """!
    Implement the DQN's value network [1] with noisy linear layers [2] from:

    [1] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves,
        Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al.
        Human-level control through deep reinforcement learning. nature, 2015.
    [2] Meire Fortunato, Mohammad Gheshlaghi Azar, Bilal Piot, Jacob Menick, Ian Osband, Alex Graves, Vlad Mnih,
        Rémi Munos, Demis Hassabis, Olivier Pietquin, Charles Blundell, and Shane Legg.
        Noisy networks for exploration. CoRR, 2017. (http://arxiv.org/abs/1706.10295)
    """

    def __init__(self, n_actions : int = 18, stack_size : Optional[int] = None) -> None:
        """!
        Constructor.
        @param n_actions: the number of actions available to the agent
        @param stack_size: the number of stacked frame in each observation, if None use the configuration
        """

        # Call the parent constructor.
        super().__init__()

        ## @var stack_size
        # Number of stacked frames in each observation.
        self.stack_size = relab.config("stack_size") if stack_size is None else stack_size

        ## @var net
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
            nn.Linear(1024, n_actions)
        )

        # Initialize the weights.
        for name, param in self.named_parameters():
            if "weight" in name:
                torch.nn.init.kaiming_normal_(param, nonlinearity="leaky_relu")

    def forward(self, x : Tensor) -> Tensor:
        """!
        Perform the forward pass through the network.
        @param x: the observation
        @return the Q-values
        """
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=0)
        return self.net(x)

    def q_values(self, x : Tensor) -> Tensor:
        """!
        Compute the Q-values for each action.
        @param x: the observation
        @return the Q-values
        """
        return self(x)
