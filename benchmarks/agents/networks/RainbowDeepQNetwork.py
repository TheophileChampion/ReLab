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

    def __init__(self, n_atoms=51, n_actions=18, v_min=-10, v_max=10):
        """
        Constructor.
        :param n_atoms: the number of atoms used to approximate the distribution over returns
        :param n_actions: the number of actions available to the agent
        :param v_min: the minimum amount of returns
        :param v_max: the maximum amount of returns
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
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
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
