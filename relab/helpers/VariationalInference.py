import math
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn.functional import (binary_cross_entropy_with_logits,
                                 gumbel_softmax)


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


def categorical_kl_divergence(
    log_alpha_hat: Tensor, log_alpha: Tensor = None
) -> Tensor:
    """!
    Compute the KL-divergence between two categorical distributions.
    @param log_alpha_hat: log-probabilities of the first distribution
    @param log_alpha: log-probabilities of the second distribution
    @return the KL-divergence
    """
    if log_alpha is None:
        n = log_alpha_hat.shape[0]
        log_alpha = -torch.ones_like(log_alpha_hat) * math.log(n)
    return torch.softmax(log_alpha_hat - log_alpha, dim=1).sum(dim=1)


def sum_categorical_kl_divergences(
    log_alpha_hats: List[Tensor], log_alphas: Optional[List[Tensor]] = None
) -> Tensor:
    """!
    Compute the sum of the KL-divergences between two list of categorical distributions.
    @param log_alpha_hats: log-probabilities of the first list of distributions
    @param log_alphas: log-probabilities of the second list of distributions
    @return the sum of KL-divergences
    """

    # Ensure validity of the parameters of the second list of distributions.
    if log_alphas is None:
        log_alphas = [None] * len(log_alpha_hats)

    # Compute the sum of KL-divergences.
    sum_kl_div = None
    for log_alpha_hat, log_alpha in zip(log_alpha_hats, log_alphas):
        if sum_kl_div is None:
            sum_kl_div = categorical_kl_divergence(log_alpha_hat, log_alpha)
        else:
            sum_kl_div += categorical_kl_divergence(log_alpha_hat, log_alpha)
    return sum_kl_div


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


def concrete_reparameterization(log_alpha: Tensor, tau: float) -> Tensor:
    """!
    Implement the reparameterization trick for a categorical distribution using the concrete distribution.
    @param log_alpha: the log-probabilities of the categorical
    @param tau: the temperature of the Gumbel-softmax
    @return the sampled state
    """
    return gumbel_softmax(log_alpha, tau)


def continuous_reparameterization(gaussian_params: Tuple[Tensor, Tensor]) -> Tensor:
    """!
    Implement the reparameterization trick for a continuous latent space.
    @param gaussian_params: the mean and logarithm of the variance of the Gaussian distribution
    @return the sampled state
    """
    mean, log_var = gaussian_params
    return gaussian_reparameterization(mean, log_var)


def discrete_reparameterization(log_alphas: List[Tensor], tau: float) -> Tensor:
    """!
    Implement the reparameterization trick for a discrete latent space.
    @param log_alphas: the log-probabilities of the categorical distributions
    @param tau: the temperature of the Gumbel-softmax
    @return the sampled state
    """
    states = [concrete_reparameterization(log_alpha, tau) for log_alpha in log_alphas]
    return torch.cat(states)


def mixed_reparameterization(
    gaussian_params: Tuple[Tensor, Tensor], log_alphas: List[Tensor], tau: float
) -> Tensor:
    """!
    Implement the reparameterization trick for a categorical distribution using the concrete distribution.
    @param gaussian_params: the mean and logarithm of the variance of the Gaussian distribution
    @param log_alphas: the log-probabilities of the categorical distributions
    @param tau: the temperature of the Gumbel-softmax
    @return the sampled state
    """
    mean, log_var = gaussian_params
    states = [concrete_reparameterization(log_alpha, tau) for log_alpha in log_alphas]
    states.append(gaussian_reparameterization(mean, log_var))
    return torch.cat(states)
