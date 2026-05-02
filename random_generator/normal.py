"""
Normal Distribution Generators
================================
Three methods to generate N(mu, sigma^2) random variables:

1. NormalBoxMuller      - Box-Muller transform (generates pairs of independents)
2. NormalCLT            - Central Limit Theorem (sum of 12 uniforms)
3. NormalRejectionSampling - Generalized rejection sampling using Double Exponential


Reference:
  - Box-Muller: Slides 1, pages 33-35
  - CLT:        Slides 1, page 36
  - Rejection:  Slides 1, pages 37-40 (uses exponential + head/tail as envelope)
"""

import math
from .continuous_generator import ContinuousGenerator
from .uniform_generator import UniformGenerator
from scipy.special import ndtri  # inverse CDF of standard normal because it's QMC-compatible (1 uniform → 1 normal, no pairing) 


class Normal(ContinuousGenerator):
    """
    Abstract base for Normal distribution generators.

    Parameters
    ----------
    mu : float
        Mean of the Normal distribution.
    sigma : float
        Standard deviation (> 0).
    uniform : UniformGenerator
        Underlying uniform generator.
    """

    def __init__(self, mu: float, sigma: float, uniform: UniformGenerator):
        if sigma <= 0:
            raise ValueError("Standard deviation sigma must be strictly positive.")
        super().__init__(
            target_mean=mu,
            target_variance=sigma ** 2,
            uniform=uniform,
        )
        self._mu = mu
        self._sigma = sigma

    @property
    def mu(self) -> float:
        return self._mu

    @property
    def sigma(self) -> float:
        return self._sigma


class NormalBoxMuller(Normal):
    """
    Normal generator using the Box-Muller transform.

    Generates two independent N(0,1) values per pair of uniforms:
      U1, U2 ~ U[0,1]
      R = sqrt(-2 * ln(U1)),   Theta = 2*pi*U2
      X = R * cos(Theta)       (returned alternately)
      Y = R * sin(Theta)

    Then Z = mu + sigma * X ~ N(mu, sigma^2).
    """

    def __init__(self, mu: float, sigma: float, uniform: UniformGenerator):
        super().__init__(mu, sigma, uniform)
        self._has_spare = False
        self._spare: float = 0.0

    def generate(self) -> float:
        """Return one N(mu, sigma^2) sample using Box-Muller (pair recycling)."""
        if self._has_spare:
            self._has_spare = False
            return self._mu + self._sigma * self._spare

        # Generate a fresh pair
        u1 = self._uniform.generate()
        u2 = self._uniform.generate()

        # Avoid log(0)
        while u1 == 0.0:
            u1 = self._uniform.generate()

        r = math.sqrt(-2.0 * math.log(u1))
        theta = 2.0 * math.pi * u2

        x = r * math.cos(theta)
        y = r * math.sin(theta)

        # store the spare for next call
        self._spare = y
        self._has_spare = True

        return self._mu + self._sigma * x


class NormalCLT(Normal):
    """
    Normal generator via the Central Limit Theorem.

    Sum of 12 independent U[0,1] and subtract 6:
      X = (sum_{i=1}^{12} U_i) - 6   ~  N(0, 1)
    """

    def generate(self) -> float:
        """Return one N(mu, sigma^2) sample using the CLT approximation."""
        total = sum(self._uniform.generate() for _ in range(12))
        z = total - 6.0   # approximately N(0, 1)
        return self._mu + self._sigma * z


class NormalRejectionSampling(Normal):
    """
    Normal generator via Generalized Rejection Sampling.

    Uses the Double Exponential (Laplace) distribution as the envelope:
      - Envelope: g(x) = 0.5 * exp(-|x|)   (Double Exponential with lambda=1)
      - Constant:  a = sqrt(2*e/pi)
      - Accept if  Y <= phi(X) / (a * g(X))

    Algorithm 
      1. Generate X ~ Exponential(1) via inverse CDF: X = -ln(U1)
      2. Generate a random sign via Head or Tail: X = +/- X
      3. Generate U2 ~ U[0,1]
      4. Accept if U2 <= exp( -(|X|-1)^2 / 2 )

  
    """

    # a = sqrt(2e/pi)
    _A = math.sqrt(2.0 * math.e / math.pi)

    def generate(self) -> float:
        """Return one N(mu, sigma^2) sample using Rejection Sampling."""
        while True:
            # Step 1: simulate X ~ Exponential(1) by inverse CDF
            u1 = self._uniform.generate()
            while u1 == 0.0:
                u1 = self._uniform.generate()
            x = -math.log(u1)

            # Step 2: assign a random sign (Head or Tail)
            u_sign = self._uniform.generate()
            if u_sign < 0.5:
                x = -x

            # Step 3: acceptance test
            u2 = self._uniform.generate()
            # Accept condition: u2 <= exp(-(|x|-1)^2 / 2)
            if u2 <= math.exp(-((abs(x) - 1.0) ** 2) / 2.0):
                return self._mu + self._sigma * x


class NormalInverseCDF(Normal):
    """
    N(mu, sigma²) via transformée inverse : Z = Φ⁻¹(U).
    QMC-compatible : 1 uniforme → 1 normale, sans pairing.
    """
    def generate(self) -> float:
        u = self._uniform.generate()
        u = max(1e-10, min(1 - 1e-10, u))
        return self._mu + self._sigma * float(ndtri(u))