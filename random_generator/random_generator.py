"""
RandomGenerator
===============
Abstract base class for all random number generators.
Mirrors the C++ RandomGenerator class exactly.

Every generator must implement Generate().
Mean() and Variance() are estimated empirically via Monte Carlo.
TestMean() and TestVariance() validate statistical properties.
"""

from abc import ABC, abstractmethod


class RandomGenerator(ABC):
    """
    Abstract base class for all random number generators.

    Parameters
    ----------
    target_mean : float
        Theoretical mean of the distribution.
    target_variance : float
        Theoretical variance of the distribution.
    """

    def __init__(self, target_mean: float = 0.0, target_variance: float = 0.0):
        if target_variance < 0:
            raise ValueError("Variance must be non-negative.")
        self._target_mean = target_mean
        self._target_variance = target_variance

    @abstractmethod
    def generate(self) -> float:
        """Generate a single random sample from the distribution."""
        ...

    # ------------------------------------------------------------------
    # Empirical statistics (Monte Carlo estimates)
    # ------------------------------------------------------------------

    def mean(self, nb_sim: int) -> float:
        """
        Estimate the mean of the distribution via Monte Carlo.

        Parameters
        ----------
        nb_sim : int
            Number of simulations to use.

        Returns
        -------
        float
            Empirical mean over nb_sim samples.
        """
        if nb_sim <= 0:
            raise ValueError("nb_sim must be strictly positive.")
        return sum(self.generate() for _ in range(nb_sim)) / nb_sim

    def variance(self, nb_sim: int) -> float:
        """
        Estimate the variance of the distribution via Monte Carlo.

        Parameters
        ----------
        nb_sim : int
            Number of simulations to use.

        Returns
        -------
        float
            Empirical variance over nb_sim samples (unbiased estimator).
        """
        if nb_sim <= 1:
            raise ValueError("nb_sim must be strictly greater than 1 to compute variance.")
        samples = [self.generate() for _ in range(nb_sim)]
        m = sum(samples) / nb_sim
        return sum((x - m) ** 2 for x in samples) / (nb_sim - 1)

    def test_mean(self, nb_sim: int, tol: float) -> bool:
        """
        Test whether the empirical mean is within tol of the target mean.

        Parameters
        ----------
        nb_sim : int
            Number of simulations.
        tol : float
            Absolute tolerance.

        Returns
        -------
        bool
            True if |empirical_mean - target_mean| <= tol.
        """
        return abs(self.mean(nb_sim) - self._target_mean) <= tol

    def test_variance(self, nb_sim: int, tol: float) -> bool:
        """
        Test whether the empirical variance is within tol of the target variance.

        Parameters
        ----------
        nb_sim : int
            Number of simulations.
        tol : float
            Absolute tolerance.

        Returns
        -------
        bool
            True if |empirical_variance - target_variance| <= tol.
        """
        return abs(self.variance(nb_sim) - self._target_variance) <= tol

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def target_mean(self) -> float:
        return self._target_mean

    @property
    def target_variance(self) -> float:
        return self._target_variance
