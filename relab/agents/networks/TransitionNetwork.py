import torch
from relab.agents.networks.layers.DiagonalGaussian import DiagonalGaussian
from torch import Tensor, nn
from torch.nn.functional import one_hot


class ContinuousTransitionNetwork(nn.Module):
    """!
    Class implementing a transition network with continuous latent variables.
    """

    def __init__(self, n_actions: int = 18, n_continuous_vars: int = 10) -> None:
        """!
        Constructor.
        @param n_actions: the number of allowable actions
        @param n_continuous_vars: the number of continuous latent variables
        """

        # Call the parent constructor.
        super().__init__()

        # @var net
        # Transition network that predicts the next state distribution.
        self.net = nn.Sequential(
            nn.Linear(n_continuous_vars + n_actions, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            DiagonalGaussian(512, n_continuous_vars),
        )

        # @var n_actions
        # Number of allowable actions in the environment.
        self.n_actions = n_actions

    def forward(self, states: Tensor, actions: Tensor) -> Tensor:
        """!
        Perform the forward pass through the network.
        @param states: the input states
        @param actions: the input actions
        @return the mean and the log of the variance of the diagonal Gaussian
        """
        actions = one_hot(actions.to(torch.int64), self.n_actions)
        x = torch.cat((states, actions), dim=1)
        return self.net(x)
