import math

import torch
from torch.nn.functional import gumbel_softmax, binary_cross_entropy_with_logits


class VariationalInference:
    """!
    A class providing useful functions for variational inference.
    """

    @staticmethod
    def gaussian_kl_divergence(mean_hat, log_var_hat, mean=None, log_var=None, min_var=1e-3):
        """!
        Compute the KL-divergence between two Gaussian distributions.
        @param mean_hat: the mean of the first Gaussian
        @param log_var_hat: the logarithm of the variance of the first Gaussian
        @param mean: the mean of the second Gaussian
        @param log_var: the logarithm of the variance of the second Gaussian
        @param min_var: the minimal variance allowed to avoid division by zero
        @return the KL-divergence
        """

        # Initialise the mean and log variance vectors to zero, if they are not provided as parameters.
        if mean is None:
            mean = torch.zeros_like(mean_hat)
        if log_var is None:
            log_var = torch.zeros_like(log_var_hat)

        # Compute the KL-divergence
        var = log_var.exp()
        var = torch.clamp(var, min=min_var)
        kl_div = log_var - log_var_hat + torch.exp(log_var_hat - log_var) + (mean - mean_hat) ** 2 / var
        return 0.5 * kl_div.sum(dim=1)

    @staticmethod
    def categorical_kl_divergence(log_alpha_hat, log_alpha=None):
        """!
        Compute the KL-divergence between two categorical distributions.
        @param log_alpha_hat: log-probabilities of the first distribution
        @param log_alpha: log-probabilities of the second distribution
        @return the KL-divergence
        """
        if log_alpha is None:
            n = log_alpha_hat.shape[0]
            log_alpha = - torch.ones_like(log_alpha_hat) * math.log(n)
        return torch.softmax(log_alpha_hat - log_alpha, dim=1).sum(dim=1)

    @staticmethod
    def gaussian_log_likelihood(obs, mean):
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
        return - 0.5 * (n * log2pi + sum_of_squared_error)

    @staticmethod
    def bernoulli_log_likelihood(obs, alpha):
        """!
        Compute the logarithm of the likelihood assuming Bernoulli distributions over pixels.
        @param obs: the image
        @param alpha: the log-probabilities of all pixels
        @return the log-likelihood
        """
        return - binary_cross_entropy_with_logits(alpha, obs, reduction="none").sum(dim=(1, 2, 3))
        # TODO one = torch.ones_like(alpha)
        # TODO zero = torch.zeros_like(alpha)
        # TODO out = - torch.maximum(alpha, zero) + alpha * obs - torch.log(one + torch.exp(-torch.abs(alpha)))
        # TODO return out.sum(dim=(1, 2, 3))

    @staticmethod
    def gaussian_reparameterization(mean, log_var):
        """!
        Implement the reparameterization trick for a Gaussian distribution.
        @param mean: the mean of the Gaussian distribution
        @param log_var: the logarithm of the variance of the Gaussian distribution
        @return the sampled state
        """
        epsilon = torch.normal(torch.zeros_like(mean), torch.ones_like(log_var))
        return epsilon * torch.exp(0.5 * log_var) + mean

    @staticmethod
    def concrete_reparameterization(log_alpha, tau):
        """!
        Implement the reparameterization trick for a categorical distribution using the concrete distribution.
        @param log_alpha: the log-probabilities of the categorical
        @param tau: the temperature of the Gumbel-softmax
        @return the sampled state
        """
        return gumbel_softmax(log_alpha, tau)

    @staticmethod
    def continuous_reparameterization(gaussian_params, tau=0):
        """!
        Implement the reparameterization trick for a continuous latent space.
        @param gaussian_params: the mean and logarithm of the variance of the Gaussian distribution
        @param tau: the temperature of the Gumbel-softmax (unused)
        @return the sampled state
        """
        mean, log_var = gaussian_params
        return VariationalInference.gaussian_reparameterization(mean, log_var)

    @staticmethod
    def discrete_reparameterization(log_alphas, tau):
        """!
        Implement the reparameterization trick for a discrete latent space.
        @param log_alphas: the log-probabilities of the categorical distributions
        @param tau: the temperature of the Gumbel-softmax
        @return the sampled state
        """
        states = [
            VariationalInference.concrete_reparameterization(log_alpha, tau)
            for log_alpha in log_alphas
        ]
        return torch.cat(states)

    @staticmethod
    def mixed_reparameterization(gaussian_params, log_alphas, tau):
        """!
        Implement the reparameterization trick for a categorical distribution using the concrete distribution.
        @param gaussian_params: the mean and logarithm of the variance of the Gaussian distribution
        @param log_alphas: the log-probabilities of the categorical distributions
        @param tau: the temperature of the Gumbel-softmax
        @return the sampled state
        """
        mean, log_var = gaussian_params
        states = [
            VariationalInference.concrete_reparameterization(log_alpha, tau)
            for log_alpha in log_alphas
        ]
        states.append(VariationalInference.gaussian_reparameterization(mean, log_var))
        return torch.cat(states)
