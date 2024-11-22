from torch import nn
import torch
import torch.nn.functional as F
from torchrl.modules import NoisyLinear


class DeepQNetwork(nn.Module):
    """
    Implement the DQN's value network from:
    Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves,
    Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al.
    Human-level control through deep reinforcement learning. nature, 2015.
    """

    def __init__(self, n_actions=18, **_):
        """
        Constructor.
        :param n_actions: the number of actions available to the agent
        """

        # Call the parent constructor.
        super().__init__()

        # Store the number of actions.
        self.n_outputs = n_actions

        # Create the layers.
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = nn.Linear(3136, 1024)
        self.fc2 = nn.Linear(1024, n_actions)

        # Initialize the weights.
        torch.nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='leaky_relu')

    def n_actions(self):
        """
        Retrieve the number of actions available to the agent.
        :return: the number of actions
        """
        return self.n_outputs

    def forward(self, x):
        """
        Perform the forward pass through the network.
        :param x: the observation
        :return: the Q-values
        """
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=0)
        x = F.leaky_relu(self.conv1(x), 0.01)
        x = F.leaky_relu(self.conv2(x), 0.01)
        x = F.leaky_relu(self.conv3(x), 0.01)
        x = F.leaky_relu(self.fc1(x.view(x.shape[0], -1)), 0.01)
        return self.fc2(x)


class NoisyDeepQNetwork(nn.Module):
    """
    Implement the DQN's value network [1] with noisy linear layers [2] from:

    [1] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves,
        Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al.
        Human-level control through deep reinforcement learning. nature, 2015.
    [2] Meire Fortunato, Mohammad Gheshlaghi Azar, Bilal Piot, Jacob Menick, Ian Osband, Alex Graves, Vlad Mnih,
        Rémi Munos, Demis Hassabis, Olivier Pietquin, Charles Blundell, and Shane Legg.
        Noisy networks for exploration. CoRR, 2017. (http://arxiv.org/abs/1706.10295)
    """

    def __init__(self, n_actions=18, **_):
        """
        Constructor.
        :param n_actions: the number of actions available to the agent
        """

        # Call the parent constructor.
        super().__init__()

        # Store the number of actions.
        self.n_outputs = n_actions

        # Create the layers.
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = NoisyLinear(3136, 1024)
        self.fc2 = NoisyLinear(1024, n_actions)

        # Initialize the weights.
        torch.nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='leaky_relu')

    def n_actions(self):
        """
        Retrieve the number of actions available to the agent.
        :return: the number of actions
        """
        return self.n_outputs

    def forward(self, x):
        """
        Perform the forward pass through the network.
        :param x: the observation
        :return: the Q-values
        """
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=0)
        x = F.leaky_relu(self.conv1(x), 0.01)
        x = F.leaky_relu(self.conv2(x), 0.01)
        x = F.leaky_relu(self.conv3(x), 0.01)
        x = F.leaky_relu(self.fc1(x.view(x.shape[0], -1)), 0.01)
        return self.fc2(x)
