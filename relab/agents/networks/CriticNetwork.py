from torch import Tensor, nn


class LinearCriticNetwork(nn.Module):
    """!
    Class implementing a critic network with continuous latent variables.
    """

    def __init__(
        self,
        n_actions: int = 18,
        n_continuous_vars: int = 10,
        n_latents: int = 512,
        n_layers: int = 4,
    ) -> None:
        """!
        Constructor.
        @param n_actions: the number of allowable actions
        @param n_continuous_vars: the number of continuous latent variables
        @param n_latents: the number of hidden neurons in the fully connected network
        @param n_layers: the number of layers in the fully connected network (at least one)
        """

        # Call the parent constructor.
        super().__init__()

        # @var net
        # Transition network that predicts the next state distribution.
        self.net = nn.Sequential()

        # Create the sequential network.
        for i in range(n_layers):

            # Add fully connected layers.
            if i == 0:
                self.net.append(nn.Linear(n_continuous_vars, n_latents))
            elif i != n_layers - 1:
                self.net.append(nn.Linear(n_latents, n_latents))
            else:
                self.net.append(nn.Linear(n_latents, n_actions))
                break

            # Add the activation function, if needed.
            self.net.append(nn.ReLU())

        # @var n_actions
        # Number of allowable actions in the environment.
        self.n_actions = n_actions

    def forward(self, states: Tensor) -> Tensor:
        """!
        Perform the forward pass through the network.
        @param states: the input states
        @return the G-values as predicted by the critic
        """
        return self.net(states)
