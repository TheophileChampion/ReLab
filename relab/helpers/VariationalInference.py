import torch
from torch import Tensor
from torch.nn.functional import binary_cross_entropy_with_logits


def gaussian_kl_divergence(
    mean_hat: Tensor,
    log_var_hat: Tensor,
    mean: Tensor = None,
    log_var: Tensor = None,
    min_var: float = 1e-3,
) -> Tensor:
    """!
    Compute the KL-divergence between two Gaussian distributions.
    @param mean_hat: the mean of the first Gaussian
    @param log_var_hat: the logarithm of the variance of the first Gaussian
    @param mean: the mean of the second Gaussian
    @param log_var: the logarithm of the variance of the second Gaussian
    @param min_var: the minimal variance allowed to avoid division by zero
    @return the KL-divergence
    """

    # Initialise mean and log variance vectors to zero, if not provided.
    if mean is None:
        mean = torch.zeros_like(mean_hat)
    if log_var is None:
        log_var = torch.zeros_like(log_var_hat)

    # Compute the KL-divergence.
    var = log_var.exp()
    var = torch.clamp(var, min=min_var)
    kl_div = log_var - log_var_hat + torch.exp(log_var_hat - log_var)
    kl_div += (mean - mean_hat) ** 2 / var
    return 0.5 * kl_div.sum(dim=1)


def gaussian_log_likelihood(obs: Tensor, mean: Tensor) -> Tensor:
    """!
    Compute the logarithm of the likelihood assuming Gaussian distributions over pixels.
    @param obs: the image
    @param mean: the mean of the Gaussian distribution
    @return the log-likelihood
    """
    sum_of_squared_error = torch.pow(obs - mean, 2).view(obs.shape[0], -1)
    n = sum_of_squared_error.shape[1]
    sum_of_squared_error = sum_of_squared_error.sum(dim=1)
    log2pi = 1.83787706641
    return -0.5 * (n * log2pi + sum_of_squared_error)


def bernoulli_log_likelihood(obs: Tensor, alpha: Tensor) -> Tensor:
    """!
    Compute the logarithm of the likelihood assuming Bernoulli distributions over pixels.
    @param obs: the image
    @param alpha: the log-probabilities of all pixels
    @return the log-likelihood
    """
    bce = binary_cross_entropy_with_logits(alpha, obs, reduction="none")
    return -bce.sum(dim=(1, 2, 3))


def gaussian_reparameterization(mean: Tensor, log_var: Tensor) -> Tensor:
    """!
    Implement the reparameterization trick for a Gaussian distribution.
    @param mean: the mean of the Gaussian distribution
    @param log_var: the logarithm of the variance of the Gaussian distribution
    @return the sampled state
    """
    epsilon = torch.normal(torch.zeros_like(mean), torch.ones_like(log_var))
    return epsilon * torch.exp(0.5 * log_var) + mean
