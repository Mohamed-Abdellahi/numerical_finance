"""
BlackScholes2D
===============
Abstract base class for 2-asset correlated Black-Scholes.

dS1 = r1*S1*dt + sigma1*S1*dW1
dS2 = r2*S2*dt + sigma2*S2*dW2
with d<W1,W2>_t = rho * dt

Concrete subclass:
  BSMilstein2D — log-Euler scheme (bs_milstein_2d.py)
"""

import math
from abc import abstractmethod
from random_generator.random_generator import RandomGenerator
from .random_process import RandomProcess


class BlackScholes2D(RandomProcess):
    """
    Abstract base for 2-asset correlated Black-Scholes.

    Parameters
    ----------
    generator : RandomGenerator
        Produces N(0,1) samples.
    spot1, spot2 : float
        Initial prices (> 0).
    rate1, rate2 : float
        Risk-free rates.
    vol1, vol2 : float
        Volatilities (> 0).
    rho : float
        Correlation between Brownian motions, in [-1, 1].
    """

    def __init__(
        self,
        generator: RandomGenerator,
        spot1: float, spot2: float,
        rate1: float, rate2: float,
        vol1: float,  vol2: float,
        rho: float,
    ):
        super().__init__(generator, dimension=2)
        if spot1 <= 0 or spot2 <= 0:
            raise ValueError("Spots must be strictly positive.")
        if vol1 <= 0 or vol2 <= 0:
            raise ValueError("Volatilities must be strictly positive.")
        if not (-1.0 <= rho <= 1.0):
            raise ValueError("Correlation rho must be in [-1, 1].")

        self._spot1 = spot1
        self._spot2 = spot2
        self._rate1 = rate1
        self._rate2 = rate2
        self._vol1 = vol1
        self._vol2 = vol2
        self._rho = rho
        self._sqrt_one_minus_rho2 = math.sqrt(max(0.0, 1.0 - rho ** 2))

    @abstractmethod
    def simulate(self, start_time: float, end_time: float, nb_steps: int) -> None:
        ...
