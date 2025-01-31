from typing import Tuple

from torch import Tensor, nn


class DiagonalGaussian(nn.Module):
    """!
    @brief Layer predicting the mean and variance of a Gaussian with diagonal covariance matrix.
    """

    def __init__(self, input_size: int, nb_components: int) -> None:
        """!
        Constructor.
        @param input_size: size of the vector send as input of the layer
        @param nb_components: the number of components of the diagonal Gaussian
        """
        super().__init__()

        # @var mean
        # Layer predicting the mean of the Gaussian distributions.
        self.mean = nn.Linear(input_size, nb_components)

        # @var log_var
        # Layer predicting the log-variance of the Gaussian distributions.
        self.log_var = nn.Linear(input_size, nb_components)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """!
        Compute the mean and the variance of the diagonal Gaussian.
        @param x: the input vector
        @return the mean and the log of the variance of the diagonal Gaussian
        """
        return self.mean(x), self.log_var(x)
