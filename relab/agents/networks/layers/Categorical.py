from torch import nn, Tensor


class Categorical(nn.Module):
    """!
    Class implementing a network that maps a vector of size n into a vector representing the log-probabilities
    of categorical distributions.
    """

    def __init__(self, input_size : int, n_discrete_vars : int, n_discrete_vals : int) -> None:
        """!
        Constructor.
        @param input_size: size of the vector send as input of the layer
        @param n_discrete_vars: the number of discrete latent variables
        @param n_discrete_vals: the number of values taken by all the discrete latent variables,
            or a list describing the number of values taken by each discrete latent variable
        """

        # Call parent constructor.
        super().__init__()

        # Ensure that the number of discrete values is a list.
        if not isinstance(n_discrete_vals, list):
            n_discrete_vals = [n_discrete_vals] * n_discrete_vars

        ## @var n_discrete_vals
        # List containing the number of values that each discrete variable can take.
        self.n_discrete_vals = n_discrete_vals

        ## @var log_alpha
        # Layer that outputs the log-probabilities for each categorical distribution.
        self.log_alpha = nn.Sequential(
            nn.Linear(input_size, sum(self.n_discrete_vals))
        )

    def forward(self, x : Tensor) -> Tensor:
        """!
        Compute the log-probabilities of the categorical distributions.
        @param x: the input vector
        @return the log-probabilities
        """
        x = self.log_alpha(x)
        xs = []
        shift = 0
        for n_discrete_val in self.n_discrete_vals:
            xs.append(x[:, shift:shift + n_discrete_val])
            shift += n_discrete_val
        return xs
