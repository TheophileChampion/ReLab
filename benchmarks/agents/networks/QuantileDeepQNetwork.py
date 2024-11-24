from torch import nn
import torch
import torch.nn.functional as F

import benchmarks


class QuantileDeepQNetwork(nn.Module):
    """
    Implement the QR-DQN's value network from:
    Will Dabney, Mark Rowland, Marc Bellemare, and Rémi Munos.
    Distributional reinforcement learning with quantile regression.
    In Proceedings of the AAAI conference on artificial intelligence, 2018.
    """

    def __init__(self, n_atoms=21, n_actions=18):
        """
        Constructor.
        :param n_atoms: the number of atoms used to approximate the distribution over returns
        :param n_actions: the number of actions available to the agent
        """

        # Call the parent constructor.
        super().__init__()

        # Store the number of atoms and actions.
        self.n_atoms = n_atoms
        self.n_actions = n_actions

        # Store the probabilities and log-probabilities of each atom.
        self.device = benchmarks.device()
        self.probs = torch.ones([n_atoms, n_actions]) / n_atoms
        self.probs = self.probs.to(self.device)
        self.log_probs = torch.log(self.probs).to(self.device)

        # Create the layers.
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
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
        atoms = self.fc2(x).view(batch_size, self.n_atoms, self.n_actions)

        # Compute all the atom probabilities and log-probabilities.
        probs = self.probs.unsqueeze(0).repeat(batch_size, 1, 1)
        log_probs = self.log_probs.unsqueeze(0).repeat(batch_size, 1, 1)

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
