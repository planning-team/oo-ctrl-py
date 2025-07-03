import numpy as np

from typing import Dict, Optional, Any, Tuple, Union
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
    """Implementation of a Normal-Log-Normal action sampler for MPPI (namely, Log-MPPI).
    
    This class samples random control perturbations from a Normal-Log-Normal distribution.
    The samples are used to explore the action space during MPPI optimization.
    For more details, see:
    Ihab S. Mohamed et al. Autonomous Navigation of AGVs in Unknown Cluttered Environments: log-MPPI Control Strategy, 2022
    
    Args:
        std_n (Tuple[float, ...]): Standard deviations for the normal distribution
        std_ln (Tuple[float, ...]): Standard deviations for the log-normal distribution
        mu_ln (Tuple[float, ...]): Means for the log-normal distribution
    """

    def __init__(self,
                 std_n: Tuple[float, ...],
                 std_ln: Tuple[float, ...],
                 mu_ln: Tuple[float, ...]):
        super(NLNActionSampler, self).__init__()
        assert len(std_n) == len(std_ln) == len(mu_ln), \
            "std_n, std_ln, and mu_ln must have the same length"
        assert np.all(np.array(std_n) > 0), "std_n must be positive"
        assert np.all(np.array(std_ln) > 0), "std_ln must be positive"
        assert np.all(np.array(mu_ln) >= 0), "mu_ln must be non-negative"
        
        self._std_n = std_n
        self._std_ln = std_ln
        self._mu_ln = mu_ln

        std_nln = [(std_n[i] ** 2) * np.exp(2 * mu_ln[i] + 2 * std_ln[i] ** 2) for i in range(len(std_n))]
        self._sigma = np.diag(std_nln)
        self._dim = len(std_n)

    def __call__(self, 
                 n_samples: int,
                 horizon: int,
                 observation: Optional[Dict[str, Any]]) -> np.ndarray:
        samples = np.zeros((n_samples, horizon, self._dim))
        for i in range(self._dim):
            samples_n = np.random.normal(0., self._std_n[i], (n_samples, horizon))
            samples_ln = np.random.lognormal(self._mu_ln[i], self._std_ln[i], (n_samples, horizon))
            samples[:, :, i] = samples_n * samples_ln
        return samples

    @property
    def covariance_matrix(self) -> np.ndarray:
        """Covariance matrix of the Normal-Log-Normal distribution.

        Returns:
            np.ndarray: Covariance matrix
        """
        return self._sigma.copy()
