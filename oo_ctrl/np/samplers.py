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
