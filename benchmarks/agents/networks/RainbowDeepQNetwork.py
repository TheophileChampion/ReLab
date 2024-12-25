from torch import nn
import torch
import torch.nn.functional as F
from torchrl.modules import NoisyLinear

import benchmarks


class RainbowDeepQNetwork(nn.Module):
    """
    Implement the rainbow DQN's value network from:
    Matteo Hessel, Joseph Modayil, Hado Van Hasselt, Tom Schaul, Georg Ostrovski, Will Dabney, Dan Horgan, Bilal Piot,
    Mohammad Azar, and David Silver. Rainbow: Combining improvements in deep reinforcement learning.
    In Proceedings of the AAAI conference on artificial intelligence, volume 32, 2018.
    """

    def __init__(self, n_atoms=51, n_actions=18, v_min=-10, v_max=10, stack_size=None):
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
        self.atoms = torch.tensor([v_min + i * self.delta_z for i in range(self.n_atoms)])
        self.atoms = self.atoms.unsqueeze(1).repeat(1, n_actions)
        self.atoms = self.atoms.to(benchmarks.device())

        # Create the layers.
        self.stack_size = benchmarks.config("stack_size") if stack_size is None else stack_size
        self.conv1 = nn.Conv2d(self.stack_size, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = NoisyLinear(3136, 1024)
        self.fc2 = NoisyLinear(1024, 512)
        self.value = NoisyLinear(512, n_atoms)
        self.advantage = NoisyLinear(512, n_atoms * n_actions)

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

        # Forward pass through the shared encoder.
        x = F.leaky_relu(self.conv1(x), 0.01)
        x = F.leaky_relu(self.conv2(x), 0.01)
        x = F.leaky_relu(self.conv3(x), 0.01)
        x = F.leaky_relu(self.fc1(x.view(batch_size, -1)), 0.01)
        x = F.leaky_relu(self.fc2(x), 0.01)

        # Compute the Q-values.
        value = self.value(x).view(batch_size, self.n_atoms, 1).repeat(1, 1, self.n_actions)
        advantages = self.advantage(x).view(batch_size, self.n_atoms, self.n_actions)
        log_probs = value + advantages - advantages.mean(dim=2).unsqueeze(dim=2).repeat(1, 1, self.n_actions)
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


class RainbowImplicitQuantileNetwork(nn.Module):
    """
    Implement the rainbow IQN's value network from:
    Marin Toromanoff, Emilie Wirbel, and Fabien Moutarde.
    Is deep reinforcement learning really superhuman on atari? leveling the playing field.
    arXiv preprint arXiv:1908.04683, 2019.
    """

    def __init__(self, n_actions=18, n_tau=64, stack_size=None):
        """
        Constructor.
        :param n_actions: the number of actions available to the agent
        :param n_tau: the size of the tau embedding
        :param stack_size: the number of stacked frame in each observation, if None use the configuration
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
        self.stack_size = benchmarks.config("stack_size") if stack_size is None else stack_size
        self.conv1 = nn.Conv2d(self.stack_size, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = NoisyLinear(3136, 1024)
        self.fc2 = NoisyLinear(1024, 512)
        self.value = NoisyLinear(512, 1)
        self.advantage = NoisyLinear(512, n_actions)
        self.tau_fc1 = NoisyLinear(self.n_tau, 3136)

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
        :return: a tuple (returns, sampled taus)
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
            tau = torch.rand([batch_size]).unsqueeze(dim=1).to(self.device)
            taus.append(tau)
            tau = torch.concat([torch.cos(torch.pi * i * tau) for i in range(self.n_tau)], dim=1)
            tau = F.leaky_relu(self.tau_fc1(tau), 0.01)

            # Compute the Q-values.
            x_tau = F.leaky_relu(self.fc1(x * tau), 0.01)
            x_tau = F.leaky_relu(self.fc2(x_tau), 0.01)
            value = self.value(x_tau).view(batch_size, 1).repeat(1, self.n_actions)
            advantages = self.advantage(x_tau).view(batch_size, self.n_actions)
            q_values = value + advantages - advantages.mean(dim=1).unsqueeze(dim=1).repeat(1, self.n_actions)
            atoms.append(q_values.unsqueeze(dim=1))

        # Concatenate all atoms and all taus along the atoms dimension.
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
