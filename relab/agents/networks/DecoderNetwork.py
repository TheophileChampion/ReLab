from typing import Optional

import relab
import torch
from torch import Tensor, nn


class ContinuousDecoderNetwork(nn.Module):
    """!
    @brief A convolutional decoder for 84x84 images with continuous latent variables.
    """

    def __init__(
        self, n_continuous_vars: int = 10, stack_size: Optional[int] = None
    ) -> None:
        """!
        Constructor.
        @param n_continuous_vars: the number of continuous latent variables
        @param stack_size: the number of stacked frame in each observation, if None use the configuration
        """

        # Call the parent constructor.
        super().__init__()

        # @var lin_net
        # Linear layers that process the continuous latent variables.
        self.lin_net = nn.Sequential(
            nn.Linear(n_continuous_vars, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 11 * 11 * 64),
            nn.ReLU(),
        )

        # @var stack_size
        # Number of stacked frames in each observation
        self.stack_size = relab.config("stack_size", stack_size)

        # @var up_conv_net
        # Transposed convolution layers predicting the reconstructed image.
        self.up_conv_net = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, 2, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 3, 2, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, self.stack_size, 3, 1, 3, 0),
        )

    def forward(self, x: Tensor) -> Tensor:
        """!
        Perform the forward pass through the network.
        @param x: the observation
        @return the reconstructed image
        """
        x = self.lin_net(x)
        x = torch.reshape(x, (x.shape[0], 64, 11, 11))
        return self.up_conv_net(x)
