import torch
import torch.nn as nn
import torch.nn.functional as F
from torchrl.modules import NoisyLinear

from benchmarks import benchmarks


class DuelingDeepQNetwork(nn.Module):
    """
    Implement a Dueling Deep Q-Network from:
    Ziyu Wang, Tom Schaul, Matteo Hessel, Hado Hasselt, Marc Lanctot, and Nando Freitas.
    Dueling network architectures for deep reinforcement learning.
    In International conference on machine learning. PMLR, 2016.
    """

    def __init__(self, n_actions=18, stack_size=None):
        """
        Constructor.
        :param n_actions: the number of actions available to the agent
        :param stack_size: the number of stacked frame in each observation, if None use the configuration
        """

        # Call the parent constructor.
        super().__init__()

        # 3 convolutional layers.
        self.stack_size = benchmarks.config("stack_size") if stack_size is None else stack_size
        self.conv1 = nn.Conv2d(self.stack_size, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        # 2 fully-connected layers followed by value and advantage head.
        self.fc1 = nn.Linear(3136, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.value = nn.Linear(512, 1)
        self.advantage = nn.Linear(512, n_actions)

        # Initialize the weights.
        torch.nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="leaky_relu")
        torch.nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="leaky_relu")
        torch.nn.init.kaiming_normal_(self.conv3.weight, nonlinearity="leaky_relu")
        torch.nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="leaky_relu")
        torch.nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="leaky_relu")
        torch.nn.init.kaiming_normal_(self.value.weight, nonlinearity="leaky_relu")
        torch.nn.init.kaiming_normal_(self.advantage.weight, nonlinearity="leaky_relu")

    def forward(self, x, return_all=False):
        """
        Perform the forward pass through the network.
        :param x: the observations
        :param return_all: True, if the Q-value, state value and advantage must be returned, False for only the Q-value
        :return: a 3-tuple (Q-values, state values, advantages) or simply the Q-values
        """

        # Forward pass through the shared encoder.
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=0)
        x = F.leaky_relu(self.conv1(x), 0.01)
        x = F.leaky_relu(self.conv2(x), 0.01)
        x = F.leaky_relu(self.conv3(x), 0.01)
        x = F.leaky_relu(self.fc1(x.view(x.size()[0], -1)), 0.01)
        x = F.leaky_relu(self.fc2(x), 0.01)

        # Compute the Q-values.
        value = self.value(x)
        advantages = self.advantage(x)
        q_values = value + advantages - advantages.mean()
        return (q_values, value, advantages) if return_all is True else q_values

    def q_values(self, x):
        """
        Compute the Q-values for each action.
        :param x: the observation
        :return: the Q-values
        """
        return self(x)


class NoisyDuelingDeepQNetwork(nn.Module):
    """
    Implement a Dueling Deep Q-Network [1] with noisy linear layers [2] from:

    [1] Ziyu Wang, Tom Schaul, Matteo Hessel, Hado Hasselt, Marc Lanctot, and Nando Freitas.
        Dueling network architectures for deep reinforcement learning.
        In International conference on machine learning. PMLR, 2016.
    [2] Meire Fortunato, Mohammad Gheshlaghi Azar, Bilal Piot, Jacob Menick, Ian Osband, Alex Graves, Vlad Mnih,
        Rémi Munos, Demis Hassabis, Olivier Pietquin, Charles Blundell, and Shane Legg.
        Noisy networks for exploration. CoRR, 2017. (http://arxiv.org/abs/1706.10295)
    """

    def __init__(self, n_actions=18, stack_size=None):
        """
        Constructor.
        :param n_actions: the number of actions available to the agent
        :param stack_size: the number of stacked frame in each observation, if None use the configuration
        """

        # Call the parent constructor.
        super().__init__()

        # 3 convolutional layers.
        self.stack_size = benchmarks.config("stack_size") if stack_size is None else stack_size
        self.conv1 = nn.Conv2d(self.stack_size, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        # 2 fully-connected layers followed by value and advantage head.
        self.fc1 = NoisyLinear(3136, 1024)
        self.fc2 = NoisyLinear(1024, 512)
        self.value = NoisyLinear(512, 1)
        self.advantage = NoisyLinear(512, n_actions)

        # Initialize the weights.
        torch.nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="leaky_relu")
        torch.nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="leaky_relu")
        torch.nn.init.kaiming_normal_(self.conv3.weight, nonlinearity="leaky_relu")
        torch.nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="leaky_relu")
        torch.nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="leaky_relu")
        torch.nn.init.kaiming_normal_(self.value.weight, nonlinearity="leaky_relu")
        torch.nn.init.kaiming_normal_(self.advantage.weight, nonlinearity="leaky_relu")

    def forward(self, x, return_all=False):
        """
        Perform the forward pass through the network.
        :param x: the observations
        :param return_all: True, if the Q-value, state value and advantage must be returned, False for only the Q-value
        :return: a tuple containing the state value and advantages
        :return: a 3-tuple (Q-values, state values, advantages) or simply the Q-values
        """

        # Forward pass through the shared encoder.
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=0)
        x = F.leaky_relu(self.conv1(x), 0.01)
        x = F.leaky_relu(self.conv2(x), 0.01)
        x = F.leaky_relu(self.conv3(x), 0.01)
        x = F.leaky_relu(self.fc1(x.view(x.size()[0], -1)), 0.01)
        x = F.leaky_relu(self.fc2(x), 0.01)

        # Compute the Q-values.
        value = self.value(x)
        advantages = self.advantage(x)
        q_values = value + advantages - advantages.mean(dim=1).unsqueeze(dim=1).repeat(1, self.n_actions)
        return (q_values, value, advantages) if return_all is True else q_values

    def q_values(self, x):
        """
        Compute the Q-values for each action.
        :param x: the observation
        :return: the Q-values
        """
        return self(x)
