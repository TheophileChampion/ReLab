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
        return self.fc2(x).view(batch_size, self.n_atoms, self.n_actions)

    def q_values(self, x):
        """
        Compute the Q-values for each action.
        :param x: the observation
        :return: the Q-values
        """
        return self(x).sum(dim=1) / self.n_atoms


class ImplicitQuantileNetwork(nn.Module):
    """
    Implement the IQN's value network from:
    Will Dabney, Georg Ostrovski, David Silver, and Rémi Munos.
    Implicit quantile networks for distributional reinforcement learning.
    In International conference on machine learning, pages 1096–1105. PMLR, 2018.
    """

    def __init__(self, n_actions=18, n_tau=64):
        """
        Constructor.
        :param n_actions: the number of actions available to the agent
        :param n_tau: the size of the tau embedding
        """

        # Call the parent constructor.
        super().__init__()

        # Store the device.
        self.device = benchmarks.device()

        # Store the number of actions and the size of the tau embedding.
        self.n_actions = n_actions
        self.n_tau = n_tau

        # Store the output of the convolutional layers.
        self.conv_output = None

        # Create the layers.
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = nn.Linear(3136, 1024)
        self.fc2 = nn.Linear(1024, n_actions)
        self.tau_fc1 = nn.Linear(self.n_tau, 3136)

        # Initialize the weights.
        torch.nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="leaky_relu")
        torch.nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="leaky_relu")
        torch.nn.init.kaiming_normal_(self.conv3.weight, nonlinearity="leaky_relu")
        torch.nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="leaky_relu")
        torch.nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="leaky_relu")
        torch.nn.init.kaiming_normal_(self.tau_fc1.weight, nonlinearity="leaky_relu")

    def compute_conv_output(self, x, invalidate_cache=False):
        """
        Compute the output of the convolutional layers.
        :param x: the observation
        :param invalidate_cache: False if the cached output should be used, True to recompute the cached output
        :return: the output of the convolutional layers
        """

        # Check if the cached convolutional output should be returned.
        if invalidate_cache is False:
            return self.conv_output

        # Compute forward pass through the convolutional layers.
        x = F.leaky_relu(self.conv1(x), 0.01)
        x = F.leaky_relu(self.conv2(x), 0.01)
        self.conv_output = F.leaky_relu(self.conv3(x), 0.01).view(x.shape[0], -1)
        return self.conv_output

    def forward(self, x, n_samples=8, invalidate_cache=True):
        """
        Perform the forward pass through the network.
        :param x: the observation
        :param n_samples: the number of taus to sample
        :param invalidate_cache: False if the cached output should be used, True to recompute the convolutional output
        :return: a 3-tuple (returns, probabilities, log-probabilities)
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
        for j in range(n_samples):

            # Compute tau embeddings.
            tau = torch.rand([batch_size])
            tau = torch.concat([
                torch.cos(torch.pi * i * tau).unsqueeze(dim=1) for i in range(self.n_tau)
            ], dim=1).to(self.device)
            tau = F.leaky_relu(self.tau_fc1(tau), 0.01)
            taus.append(tau)

            # Compute the output
            x_tau = F.leaky_relu(self.fc1(x * tau), 0.01)
            atoms_tau = self.fc2(x_tau).view(batch_size, 1, self.n_actions)
            atoms.append(atoms_tau)

        # Concatenate all atoms along the second dimension, i.e., atoms dimension.
        return torch.concat(atoms, dim=1), torch.concat(taus, dim=1)

    def q_values(self, x, n_samples=32, invalidate_cache=True):
        """
        Compute the Q-values for each action.
        :param x: the observation
        :param n_samples: the number of samples used to estimate the Q-values
        :param invalidate_cache: False if the cached output should be used, True to recompute the convolutional output
        :return: the Q-values
        """
        atoms, _ = self(x, n_samples=n_samples, invalidate_cache=invalidate_cache)
        return atoms.sum(dim=1) / n_samples
