from typing import List, Tuple, Optional

from torch import nn, Tensor
import torch
from math import prod

from relab import relab
from relab.agents.networks.layers.Categorical import Categorical
from relab.agents.networks.layers.DiagonalGaussian import DiagonalGaussian


class ContinuousEncoderNetwork(nn.Module):
    """!
    Class implementing a convolutional encoder for 84 by 84 images with continuous latent variables.
    """

    def __init__(self, n_continuous_vars : int = 10, stack_size : Optional[int] = None) -> None:
        """!
        Constructor.
        @param n_continuous_vars: the number of continuous latent variables
        @param stack_size: the number of stacked frame in each observation, if None use the configuration
        """

        # Call the parent constructor.
        super().__init__()

        ## @var stack_size
        # Number of stacked frames in each observation.
        self.stack_size = 1
        # TODO self.stack_size = relab.config("stack_size") if stack_size is None else stack_size

        ## @var conv_net
        # Convolutional encoder network that processes the input images.
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

        ## @var conv_output_shape
        # Shape of the features output by the convolutional encoder network.
        self.conv_output_shape = self.conv_output_shape([self.stack_size, 84, 84])
        self.conv_output_shape = self.conv_output_shape[1:]
        conv_output_size = prod(self.conv_output_shape)

        ## @var linear_net
        # Linear encoder network that processes flattened convolutional features.
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

        ## @var net
        # Complete encoder network combining convolutional and linear parts.
        self.net = nn.Sequential(
            self.conv_net,
            self.linear_net
        )

    def conv_output_shape(self, image_shape : List[int]) -> torch.Size:
        """!
        Compute the shape of the features output by the convolutional encoder.
        @param image_shape: the shape of the input image
        @return the shape of the features output by the convolutional encoder
        """
        # TODO test and remove? image_shape = list(image_shape)
        image_shape.insert(0, 1)
        input_image = torch.zeros(image_shape)
        return self.conv_net(input_image).shape

    def forward(self, x : Tensor) -> Tensor:
        """!
        Perform the forward pass through the network.
        @param x: the observation
        @return the mean and the log of the variance of the diagonal Gaussian
        """
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=0)
        return self.net(x)


class DiscreteEncoderNetwork(nn.Module):
    """!
    Class implementing a convolutional encoder for 84 by 84 images with discrete latent variables.
    """

    def __init__(
        self,
        n_discrete_vars : int = 20,
        n_discrete_vals : int = 10,
        stack_size : Optional[int] = None
    ) -> None:
        """!
        Constructor.
        @param n_discrete_vars: the number of discrete latent variables
        @param n_discrete_vals: the number of values taken by all the discrete latent variables,
            or a list describing the number of values taken by each discrete latent variable
        @param stack_size: the number of stacked frame in each observation, if None use the configuration
        """

        # Call the parent constructor.
        super().__init__()

        ## @var stack_size
        # Number of stacked frames in each observation.
        self.stack_size = relab.config("stack_size") if stack_size is None else stack_size

        ## @var conv_net
        # Convolutional encoder network that processes the input images.
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

        ## @var conv_output_shape
        # Shape of the features output by the convolutional encoder network.
        self.conv_output_shape = self.conv_output_shape([self.stack_size, 84, 84])
        self.conv_output_shape = self.conv_output_shape[1:]
        conv_output_size = prod(self.conv_output_shape)

        ## @var linear_net
        # Linear encoder network that processes flattened convolutional features.
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

        ## @var net
        # Complete encoder network combining convolutional and linear parts.
        self.net = nn.Sequential(
            self.conv_net,
            self.linear_net
        )

    def conv_output_shape(self, image_shape : List[int]) -> torch.Size:
        """!
        Compute the shape of the features output by the convolutional encoder.
        @param image_shape: the shape of the input image
        @return the shape of the features output by the convolutional encoder
        """
        # TODO test and remove? image_shape = list(image_shape)
        image_shape.insert(0, 1)
        input_image = torch.zeros(image_shape)
        return self.conv_net(input_image).shape

    def forward(self, x : Tensor) -> Tensor:
        """!
        Perform the forward pass through the network.
        @param x: the observation
        @return the mean and the log of the variance of the diagonal Gaussian
        """
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=0)
        return self.net(x)


class MixedEncoderNetwork(nn.Module):
    """!
    Class implementing a convolutional encoder for 84 by 84 images with discrete and continuous latent variables.
    """

    def __init__(
        self,
        n_continuous_vars : int = 10,
        n_discrete_vars : int = 20,
        n_discrete_vals : int = 10,
        stack_size : Optional[int] = None
    ) -> None:
        """!
        Constructor.
        @param n_continuous_vars: the number of continuous latent variables
        @param n_discrete_vars: the number of discrete latent variables
        @param n_discrete_vals: the number of values taken by all the discrete latent variables,
            or a list describing the number of values taken by each discrete latent variable
        @param stack_size: the number of stacked frame in each observation, if None use the configuration
        """

        # Call the parent constructor.
        super().__init__()

        ## @var stack_size
        # Number of stacked frames in each observation.
        self.stack_size = relab.config("stack_size") if stack_size is None else stack_size

        ## @var conv_net
        # Convolutional encoder network that processes the input images.
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

        ## @var conv_output_shape
        # Shape of the features output by the convolutional encoder network.
        self.conv_output_shape = self.conv_output_shape([self.stack_size, 84, 84])
        self.conv_output_shape = self.conv_output_shape[1:]
        conv_output_size = prod(self.conv_output_shape)

        ## @var linear_net
        # Linear encoder network that processes flattened convolutional features.
        self.linear_net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(conv_output_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        ## @var net
        # Complete encoder network combining convolutional and linear parts.
        self.net = nn.Sequential(
            self.conv_net,
            self.linear_net
        )

        ## @var gaussian_head
        # Network head that outputs the mean and log variance of the continuous latent variables.
        self.gaussian_head = DiagonalGaussian(256, n_continuous_vars)

        ## @var categorical_head
        # Network head that outputs the log-probabilities of the discrete latent variables.
        self.categorical_head = Categorical(256, n_discrete_vars, n_discrete_vals)

    def conv_output_shape(self, image_shape : List[int]) -> torch.Size:
        """!
        Compute the shape of the features output by the convolutional encoder.
        @param image_shape: the shape of the input image
        @return the shape of the features output by the convolutional encoder
        """
        # TODO test and remove? image_shape = list(image_shape)
        image_shape.insert(0, 1)
        input_image = torch.zeros(image_shape)
        return self.conv_net(input_image).shape

    def forward(self, x : Tensor) -> Tuple[Tensor, Tensor]:
        """!
        Perform the forward pass through the network.
        @param x: the observation
        @return the mean and the log of the variance of the diagonal Gaussian
        """
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=0)
        x = self.net(x)
        return self.gaussian_head(x), self.categorical_head(x)
