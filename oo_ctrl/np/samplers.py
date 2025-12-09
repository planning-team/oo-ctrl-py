import numpy as np

from typing import Dict, Optional, Any, Tuple
from oo_ctrl.np.core import AbstractActionSampler


class GaussianActionSampler(AbstractActionSampler):
    """Implementation of a Gaussian action sampler for MPPI.

    This class samples random control perturbations from a multivariate Gaussian distribution
    with zero mean and specified covariance matrix. The samples are used to explore the action
    space during MPPI optimization.

    Args:
        stds (Tuple[float, ...]): Standard deviations for each control dim to build a covariance matrix
    """

    def __init__(self, 
                 stds: Tuple[float, ...]):
        super(GaussianActionSampler, self).__init__()
        self._sigma = np.diag(stds) ** 2

    def __call__(self, 
                 n_samples: int,
                 horizon: int,
                 observation: Optional[Dict[str, Any]]) -> np.ndarray:
        mu = np.zeros(self._sigma.shape[0])
        return np.random.multivariate_normal(mu, self._sigma, (n_samples, horizon))

    @property
    def covariance_matrix(self) -> np.ndarray:
        """Covariance matrix of the Gaussian distribution.

        Returns:
            np.ndarray: Covariance matrix
        """
        return self._sigma.copy()


class NLNActionSampler(AbstractActionSampler):

    def __init__(self,
                 stds: Tuple[float, ...]) -> None:
        super(NLNActionSampler, self).__init__()
        # Variance of the normal distribution
        var_n = np.array(stds) ** 2
        mu_n = np.zeros(len(stds))
        # Mean and variance of the log-normal distribution
        mu_ln = np.exp(0.5 * var_n)
        var_ln = np.exp(var_n) * (np.exp(var_n) - 1.)
        
        var_nln = var_n * np.exp(2 * mu_ln + 2 * var_ln)

        self._dim = len(stds)
        self._mu_n = mu_n
        self._std_n = np.sqrt(var_n)
        self._mu_ln = mu_ln
        self._std_ln = np.sqrt(var_ln)
        self._covariance_matrix = np.diag(var_nln)

    def __call__(self, 
                 n_samples: int,
                 horizon: int,
                 observation: Optional[Dict[str, Any]]) -> np.ndarray:
        samples_n = np.random.normal(self._mu_n, self._std_n, (n_samples, horizon, self._dim))
        samples_ln = np.random.lognormal(self._mu_ln, self._std_ln, (n_samples, horizon, self._dim))
        samples = samples_n * samples_ln
        return samples

    @property
    def covariance_matrix(self) -> np.ndarray:
        return self._covariance_matrix.copy()
