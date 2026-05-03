"""
Exponential Distribution Generators
=====================================
Two algorithms to generate Exp(lambda) random variables:

1. ExponentialInverseDistribution - Inverse CDF method: X = -ln(U) / lambda
2. ExponentialRejectionSampling   - Rejection sampling on [0, b] with uniform envelope

"""

import math
from .continuous_generator import ContinuousGenerator
from .uniform_generator import UniformGenerator


class Exponential(ContinuousGenerator):
    """
    Abstract base for Exponential(lambda) generators.
    Mean = 1/lambda, Variance = 1/lambda^2.

    Parameters
    ----------
    lam : float
        Rate parameter (lambda > 0).
    uniform : UniformGenerator
        Underlying uniform generator.
    """

    def __init__(self, lam: float, uniform: UniformGenerator):
        if lam <= 0:
            raise ValueError("Rate parameter lambda must be strictly positive.")
        super().__init__(
            target_mean=1.0 / lam,
            target_variance=1.0 / (lam ** 2),
            uniform=uniform,
        )
        self._lam = lam

    @property
    def lam(self) -> float:
        return self._lam


class ExponentialInverseDistribution(Exponential):
    """
    Exponential generator using the Inverse CDF (quantile) method.

    F^{-1}(u) = -ln(1 - u) / lambda  ≈  -ln(u) / lambda
    """

    def generate(self) -> float:
        """Return one Exp(lambda) sample via inverse CDF."""
        u = self._uniform.generate()
        while u == 0.0:
            u = self._uniform.generate()
        return -math.log(u) / self._lam


class ExponentialRejectionSampling(Exponential):
    """
    Exponential generator using Rejection Sampling on [0, b].

    Envelope: M * uniform on [0, b]  where  M = b * lambda * exp(-lambda * b) ... wait,
    we use the simplest approach: on [0, b], the exponential density is bounded
    by f(0) = lambda. So the envelope is M = lambda, g=Uniform[0,b].

    Algorithm:
      1. Generate X ~ U[0, b]
      2. Generate Y ~ U[0, M]  (M = lambda)
      3. Accept if Y <= f(X) = lambda * exp(-lambda * X)

    Note: b should be chosen large enough that F(b) ≈ 1 (e.g. b = 10/lambda).

    """

    def __init__(self, lam: float, uniform: UniformGenerator, b: float = None):
        super().__init__(lam, uniform)
        # b is the upper bound of the interval; default to 10/lambda
        self._b = b if b is not None else 10.0 / lam
        self._M = lam  # Upper bound of the density on [0, b]

    def generate(self) -> float:
        """Return one Exp(lambda) sample via Rejection Sampling on [0, b]."""
        while True:
            x = self._uniform.generate() * self._b      # X ~ U[0, b]
            y = self._uniform.generate() * self._M      # Y ~ U[0, M]
            fx = self._lam * math.exp(-self._lam * x)   # f(x) = lambda * exp(-lambda*x)
            if y <= fx:
                return x
