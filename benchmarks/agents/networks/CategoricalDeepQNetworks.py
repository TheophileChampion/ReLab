from torch import nn
import torch
import torch.nn.functional as F
from torchrl.modules import NoisyLinear

import benchmarks


class CategoricalDeepQNetwork(nn.Module):
    """
    Implement the categorical DQN's value network from:
    Marc G Bellemare, Will Dabney, and Rémi Munos. A distributional perspective on reinforcement learning.
    In International conference on machine learning. PMLR, 2017.
    """

    def __init__(self, n_atoms=21, n_actions=18, v_min=-10, v_max=10, stack_size=None):
        """
        Constructor.
        :param n_atoms: the number of atoms used to approximate the distribution over returns
        :param n_actions: the number of actions available to the agent
        :param v_min: the minimum amount of returns
        :param v_max: the maximum amount of returns
        :param stack_size: the number of stacked frame in each observation, if None use the configuration
        """

        # Call the parent constructor.
        super().__init__()

        # Store the number of atoms and actions.
        self.n_atoms = n_atoms
        self.n_actions = n_actions

        # Store the minimum and maximum amount of returns.
        self.v_min = v_min
        self.v_max = v_max

        # Compute the atoms (canonical returns).
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        self.atoms = torch.arange(v_min, v_max + 1, self.delta_z)
        self.atoms = self.atoms.unsqueeze(1).repeat(1, n_actions)
        self.atoms = self.atoms.to(benchmarks.device())

        # Create the layers.
        self.stack_size = benchmarks.config("stack_size") if stack_size is None else stack_size
        self.conv1 = nn.Conv2d(self.stack_size, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = nn.Linear(3136, 1024)
        self.fc2 = nn.Linear(1024, n_atoms * n_actions)

        # Initialize the weights.
        torch.nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="leaky_relu")
        torch.nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="leaky_relu")
        torch.nn.init.kaiming_normal_(self.conv3.weight, nonlinearity="leaky_relu")
        torch.nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="leaky_relu")
        torch.nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="leaky_relu")

    def forward(self, x):
        """
        Perform the forward pass through the network.
        :param x: the observation
        :return: a 3-tuple (returns, probabilities, log-probabilities)
        """

        # Ensure the input has the correct shape.
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=0)
        batch_size = x.shape[0]

        # Compute forward pass.
        x = F.leaky_relu(self.conv1(x), 0.01)
        x = F.leaky_relu(self.conv2(x), 0.01)
        x = F.leaky_relu(self.conv3(x), 0.01)
        x = F.leaky_relu(self.fc1(x.view(batch_size, -1)), 0.01)
        log_probs = self.fc2(x).view(batch_size, self.n_atoms, self.n_actions)
        probs = log_probs.softmax(dim=1)

        # Compute all atoms.
        atoms = self.atoms.unsqueeze(0).repeat(batch_size, 1, 1)

        # Return all atoms, their probabilities and log-probabilities.
        return atoms, probs, log_probs

    def q_values(self, x):
        """
        Compute the Q-values for each action.
        :param x: the observation
        :return: the Q-values
        """
        atoms, probs, _ = self(x)
        return (atoms * probs).sum(dim=1)


class NoisyCategoricalDeepQNetwork(nn.Module):
    """
    Implement the categorical DQN's value network [1] with noisy linear layers [2] from:

    [1] Marc G Bellemare, Will Dabney, and Rémi Munos. A distributional perspective on reinforcement learning.
        In International conference on machine learning. PMLR, 2017.
    [2] Meire Fortunato, Mohammad Gheshlaghi Azar, Bilal Piot, Jacob Menick, Ian Osband, Alex Graves, Vlad Mnih,
        Rémi Munos, Demis Hassabis, Olivier Pietquin, Charles Blundell, and Shane Legg.
        Noisy networks for exploration. CoRR, 2017. (http://arxiv.org/abs/1706.10295)
    """

    def __init__(self, n_atoms=21, n_actions=18, v_min=-10, v_max=10, stack_size=None):
        """
        Constructor.
        :param n_atoms: the number of atoms used to approximate the distribution over returns
        :param n_actions: the number of actions available to the agent
        :param v_min: the minimum amount of returns
        :param v_max: the maximum amount of returns
        :param stack_size: the number of stacked frame in each observation, if None use the configuration
        """

        # Call the parent constructor.
        super().__init__()

        # Store the number of atoms and actions.
        self.n_atoms = n_atoms
        self.n_actions = n_actions

        # Store the minimum and maximum amount of returns.
        self.v_min = v_min
        self.v_max = v_max

        # Compute the atoms (canonical returns).
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        self.atoms = torch.arange(v_min, v_max + 1, self.delta_z)
        self.atoms = self.atoms.unsqueeze(1).repeat(1, n_actions)
        self.atoms = self.atoms.to(benchmarks.device())

        # Create the layers.
        self.stack_size = benchmarks.config("stack_size") if stack_size is None else stack_size
        self.conv1 = nn.Conv2d(self.stack_size, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = NoisyLinear(3136, 1024)
        self.fc2 = NoisyLinear(1024, n_atoms * n_actions)

        # Initialize the weights.
        torch.nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="leaky_relu")
        torch.nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="leaky_relu")
        torch.nn.init.kaiming_normal_(self.conv3.weight, nonlinearity="leaky_relu")
        torch.nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="leaky_relu")
        torch.nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="leaky_relu")

    def forward(self, x):
        """
        Perform the forward pass through the network.
        :param x: the observation
        :return: a 3-tuple (returns, probabilities, log-probabilities)
        """

        # Ensure the input has the correct shape.
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=0)
        batch_size = x.shape[0]

        # Compute forward pass.
        x = F.leaky_relu(self.conv1(x), 0.01)
        x = F.leaky_relu(self.conv2(x), 0.01)
        x = F.leaky_relu(self.conv3(x), 0.01)
        x = F.leaky_relu(self.fc1(x.view(batch_size, -1)), 0.01)
        log_probs = self.fc2(x).view(batch_size, self.n_atoms, self.n_actions)
        probs = log_probs.softmax(dim=1)

        # Compute all atoms.
        atoms = self.atoms.unsqueeze(0).repeat(batch_size, 1, 1)

        # Return all atoms, their probabilities and log-probabilities.
        return atoms, probs, log_probs

    def q_values(self, x):
        """
        Compute the Q-values for each action.
        :param x: the observation
        :return: the Q-values
        """
        atoms, probs, _ = self(x)
        return (atoms * probs).sum(dim=1)
