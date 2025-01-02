from torch import nn
import torch
from math import prod

from benchmarks import benchmarks
from benchmarks.agents.networks.layers.Categorical import Categorical
from benchmarks.agents.networks.layers.DiagonalGaussian import DiagonalGaussian


class ContinuousEncoderNetwork(nn.Module):
    """
    Class implementing a convolutional encoder for 84 by 84 images with continuous latent variables.
    """

    def __init__(self, n_continuous_vars=10, stack_size=None):
        """
        Constructor.
        :param n_continuous_vars: the number of continuous latent variables
        :param stack_size: the number of stacked frame in each observation, if None use the configuration
        """

        # Call the parent constructor.
        super().__init__()

        # Create the convolutional encoder network.
        # TODO self.stack_size = benchmarks.config("stack_size") if stack_size is None else stack_size
        self.stack_size = 1
        self.conv_net = nn.Sequential(
            nn.Conv2d(self.stack_size, 32, (3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), stride=(2, 2), padding=1),
            nn.ReLU()
        )
        self.conv_output_shape = self.conv_output_shape([self.stack_size, 84, 84])
        self.conv_output_shape = self.conv_output_shape[1:]
        conv_output_size = prod(self.conv_output_shape)

        # Create the linear encoder network.
        self.linear_net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(conv_output_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            DiagonalGaussian(256, n_continuous_vars)
        )

        # Create the full encoder network.
        self.net = nn.Sequential(
            self.conv_net,
            self.linear_net
        )

    def conv_output_shape(self, image_shape):
        """
        Compute the shape of the features output by the convolutional encoder.
        :param image_shape: the shape of the input image
        :return: the shape of the features output by the convolutional encoder
        """
        image_shape = list(image_shape)
        image_shape.insert(0, 1)
        input_image = torch.zeros(image_shape)
        return self.conv_net(input_image).shape

    def forward(self, x):
        """
        Perform the forward pass through the network.
        :param x: the observation
        :return: the mean and the log of the variance of the diagonal Gaussian
        """
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=0)
        return self.net(x)


class DiscreteEncoderNetwork(nn.Module):
    """
    Class implementing a convolutional encoder for 84 by 84 images with discrete latent variables.
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

        # Create the convolutional encoder network.
        self.stack_size = benchmarks.config("stack_size") if stack_size is None else stack_size
        self.conv_net = nn.Sequential(
            nn.Conv2d(self.stack_size, 32, (3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), stride=(2, 2), padding=1),
            nn.ReLU()
        )
        self.conv_output_shape = self.conv_output_shape([self.stack_size, 84, 84])
        self.conv_output_shape = self.conv_output_shape[1:]
        conv_output_size = prod(self.conv_output_shape)

        # Create the linear encoder network.
        self.linear_net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(conv_output_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            Categorical(256, n_discrete_vars, n_discrete_vals)
        )

        # Create the full encoder network.
        self.net = nn.Sequential(
            self.conv_net,
            self.linear_net
        )

    def conv_output_shape(self, image_shape):
        """
        Compute the shape of the features output by the convolutional encoder.
        :param image_shape: the shape of the input image
        :return: the shape of the features output by the convolutional encoder
        """
        image_shape = list(image_shape)
        image_shape.insert(0, 1)
        input_image = torch.zeros(image_shape)
        return self.conv_net(input_image).shape

    def forward(self, x):
        """
        Perform the forward pass through the network.
        :param x: the observation
        :return: the mean and the log of the variance of the diagonal Gaussian
        """
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=0)
        return self.net(x)


class MixedEncoderNetwork(nn.Module):
    """
    Class implementing a convolutional encoder for 84 by 84 images with discrete and continuous latent variables.
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

        # Create the convolutional encoder network.
        self.stack_size = benchmarks.config("stack_size") if stack_size is None else stack_size
        self.conv_net = nn.Sequential(
            nn.Conv2d(self.stack_size, 32, (3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), stride=(2, 2), padding=1),
            nn.ReLU()
        )
        self.conv_output_shape = self.conv_output_shape([self.stack_size, 84, 84])
        self.conv_output_shape = self.conv_output_shape[1:]
        conv_output_size = prod(self.conv_output_shape)

        # Create the linear encoder network.
        self.linear_net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(conv_output_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # Create the full encoder network and the network heads.
        self.net = nn.Sequential(
            self.conv_net,
            self.linear_net
        )
        self.gaussian_head = DiagonalGaussian(256, n_continuous_vars)
        self.categorical_head = Categorical(256, n_discrete_vars, n_discrete_vals)

    def conv_output_shape(self, image_shape):
        """
        Compute the shape of the features output by the convolutional encoder.
        :param image_shape: the shape of the input image
        :return: the shape of the features output by the convolutional encoder
        """
        image_shape = list(image_shape)
        image_shape.insert(0, 1)
        input_image = torch.zeros(image_shape)
        return self.conv_net(input_image).shape

    def forward(self, x):
        """
        Perform the forward pass through the network.
        :param x: the observation
        :return: the mean and the log of the variance of the diagonal Gaussian
        """
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=0)
        x = self.net(x)
        return self.gaussian_head(x), self.categorical_head(x)
