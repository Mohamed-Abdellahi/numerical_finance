"""
Poisson Distribution Generator
================================
Two algorithms to generate Poisson(lambda) random variables.

Algorithm 1 (Inverse CDF / direct):
  Use the relation: P(N=k) = exp(-lambda) * lambda^k / k!
  Simulate by comparing a uniform with cumulative probabilities.

Algorithm 2 (via Exponential inter-arrivals):
  Count arrivals in [0,1]: N = max{k : E1+...+Ek <= lambda}
  where each Ei ~ Exp(1). Equivalently: N = max{k : U1*...*Uk >= exp(-lambda)}.

Reference: Slides 1 (Kaiza Amouh), Generator Architecture PDF
"""

import math
import enum
from .continuous_generator import ContinuousGenerator
from .uniform_generator import UniformGenerator


class PoissonAlgo(enum.Enum):
    """Algorithm selection for the Poisson generator."""
    INVERSE_CDF = "inverse_cdf"          # Algorithm 1: cumulative sum of probabilities
    EXPONENTIAL_ARRIVALS = "exp_arrivals" # Algorithm 2: product of uniforms


class Poisson(ContinuousGenerator):
    """
    Poisson(lambda) generator.
    Mean = lambda, Variance = lambda.

    Parameters
    ----------
    lam : float
        Rate parameter (lambda > 0).
    uniform : UniformGenerator
        Underlying uniform generator.
    algo : PoissonAlgo
        Which simulation algorithm to use.
    """

    def __init__(
        self,
        lam: float,
        uniform: UniformGenerator,
        algo: PoissonAlgo = PoissonAlgo.EXPONENTIAL_ARRIVALS,
    ):
        if lam <= 0:
            raise ValueError("Rate parameter lambda must be strictly positive.")
        super().__init__(
            target_mean=lam,
            target_variance=lam,
            uniform=uniform,
        )
        self._lam = lam
        self._algo = algo
        # Precompute exp(-lambda) for efficiency (Algorithm 2)
        self._exp_neg_lam = math.exp(-lam)

    def generate(self) -> float:
        """Return one Poisson(lambda) sample (returned as float for API consistency)."""
        if self._algo == PoissonAlgo.INVERSE_CDF:
            return float(self._generate_inverse_cdf())
        else:
            return float(self._generate_exp_arrivals())

    def _generate_inverse_cdf(self) -> int:
        """
        Algorithm 1: Inverse CDF via cumulative probabilities.

        P(N=0) = exp(-lambda)
        P(N=k) = P(N=k-1) * lambda / k

        We draw U ~ U[0,1] and find the smallest k such that F(k) >= U.
        """
        u = self._uniform.generate()
        p = self._exp_neg_lam   # P(N=0)
        cdf = p
        k = 0
        while cdf < u:
            k += 1
            p *= self._lam / k
            cdf += p
        return k

    def _generate_exp_arrivals(self) -> int:
        """
        Algorithm 2: Count inter-arrival times (product of uniforms).

        Stop when U1 * U2 * ... * Uk < exp(-lambda).
        The result is k-1.
        """
        product = self._uniform.generate()
        k = 0
        while product >= self._exp_neg_lam:
            product *= self._uniform.generate()
            k += 1
        return k
