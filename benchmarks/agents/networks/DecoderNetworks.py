from torch import nn
import torch

from benchmarks import benchmarks


class ContinuousDecoderNetwork(nn.Module):
    """
    Class implementing a convolutional decoder for 84 by 84 images with continuous latent variables.
    """

    def __init__(self, n_continuous_vars=10, stack_size=None):
        """
        Constructor.
        :param n_continuous_vars: the number of continuous latent variables
        :param stack_size: the number of stacked frame in each observation, if None use the configuration
        """

        # Call the parent constructor.
        super().__init__()

        # Create the up-convolutional network.
        self.lin_net = nn.Sequential(
            nn.Linear(n_continuous_vars, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 16 * 16 * 64),
            nn.ReLU(),
        )
        self.stack_size = benchmarks.config("stack_size") if stack_size is None else stack_size
        self.up_conv_net = nn.Sequential(
            nn.ConvTranspose2d(64, 64, (3, 3), stride=(2, 2), output_padding=(0, 0)),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, (3, 3), stride=(2, 2), padding=(1, 1), output_padding=(0, 0)),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, (3, 3), stride=(2, 2), padding=(2, 2), output_padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, self.stack_size, (3, 3), stride=(1, 1), padding=(3, 3), output_padding=(0, 0))
        )

    def forward(self, x):
        """
        Perform the forward pass through the network.
        :param x: the observation
        :return: the reconstructed image
        """
        x = self.lin_net(x)
        x = torch.reshape(x, (x.shape[0], 64, 16, 16))
        return self.up_conv_net(x)


class DiscreteDecoderNetwork(nn.Module):
    """
    Class implementing a convolutional decoder for 84 by 84 images with discrete latent variables.
    """

    def __init__(self, n_discrete_vars=20, n_discrete_vals=10, stack_size=None):
        """
        Constructor.
        :param n_discrete_vars: the number of discrete latent variables
        :param n_discrete_vals: the number of values taken by all the discrete latent variables,
            or a list describing the number of values taken by each discrete latent variable
        :param stack_size: the number of stacked frame in each observation, if None use the configuration
        """

        # Call the parent constructor.
        super().__init__()

        # Create the up-convolutional network.
        self.lin_net = nn.Sequential(
            nn.Linear(n_discrete_vars * n_discrete_vals, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 16 * 16 * 64),
            nn.ReLU(),
        )
        self.stack_size = benchmarks.config("stack_size") if stack_size is None else stack_size
        self.up_conv_net = nn.Sequential(
            nn.ConvTranspose2d(64, 64, (3, 3), stride=(2, 2), output_padding=(0, 0)),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, (3, 3), stride=(2, 2), padding=(1, 1), output_padding=(0, 0)),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, (3, 3), stride=(2, 2), padding=(2, 2), output_padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, self.stack_size, (3, 3), stride=(1, 1), padding=(3, 3), output_padding=(0, 0))
        )

    def forward(self, x):
        """
        Perform the forward pass through the network.
        :param x: the observation
        :return: the reconstructed image
        """
        x = self.lin_net(x)
        x = torch.reshape(x, (x.shape[0], 64, 16, 16))
        return self.up_conv_net(x)


class MixedDecoderNetwork(nn.Module):
    """
    Class implementing a convolutional decoder for 84 by 84 images with discrete and continuous latent variables.
    """

    def __init__(self, n_continuous_vars=10, n_discrete_vars=20, n_discrete_vals=10, stack_size=None):
        """
        Constructor.
        :param n_continuous_vars: the number of continuous latent variables
        :param n_discrete_vars: the number of discrete latent variables
        :param n_discrete_vals: the number of values taken by all the discrete latent variables,
            or a list describing the number of values taken by each discrete latent variable
        :param stack_size: the number of stacked frame in each observation, if None use the configuration
        """

        # Call the parent constructor.
        super().__init__()

        # Create the up-convolutional network.
        self.lin_net = nn.Sequential(
            nn.Linear(n_continuous_vars + n_discrete_vars * n_discrete_vals, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 16 * 16 * 64),
            nn.ReLU(),
        )
        self.stack_size = benchmarks.config("stack_size") if stack_size is None else stack_size
        self.up_conv_net = nn.Sequential(
            nn.ConvTranspose2d(64, 64, (3, 3), stride=(2, 2), output_padding=(0, 0)),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, (3, 3), stride=(2, 2), padding=(1, 1), output_padding=(0, 0)),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, (3, 3), stride=(2, 2), padding=(2, 2), output_padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, self.stack_size, (3, 3), stride=(1, 1), padding=(3, 3), output_padding=(0, 0))
        )

    def forward(self, x):
        """
        Perform the forward pass through the network.
        :param x: the observation
        :return: the reconstructed image
        """
        x = self.lin_net(x)
        x = torch.reshape(x, (x.shape[0], 64, 16, 16))
        return self.up_conv_net(x)
