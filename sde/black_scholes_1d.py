"""
BlackScholes1D
===============
Abstract base class for 1-dimensional Black-Scholes diffusion.

  dS_t = r * S_t * dt + sigma * S_t * dW_t

Concrete subclasses:
  BSEuler1D    — Euler-Maruyama scheme (bs_euler_1d.py)
  BSMilstein1D — Milstein / log-Euler scheme (bs_milstein_1d.py)
"""

from abc import abstractmethod
from random_generator.random_generator import RandomGenerator
from .random_process import RandomProcess


class BlackScholes1D(RandomProcess):
    """
    Abstract base for 1D Black-Scholes processes.

    dS_t = r * S_t * dt + sigma * S_t * dW_t

    Parameters
    ----------
    generator : RandomGenerator
        Produces N(0,1) samples.
    spot : float
        Initial asset price S_0 (> 0).
    rate : float
        Risk-free interest rate r.
    vol : float
        Volatility sigma (> 0).
    """

    def __init__(
        self,
        generator: RandomGenerator,
        spot: float,
        rate: float,
        vol: float,
    ):
        super().__init__(generator, dimension=1)
        if spot <= 0:
            raise ValueError("Spot must be strictly positive.")
        if vol <= 0:
            raise ValueError("Volatility must be strictly positive.")
        self._spot = spot
        self._rate = rate
        self._vol = vol

    @abstractmethod
    def simulate(self, start_time: float, end_time: float, nb_steps: int) -> None:
        ...

    @property
    def spot(self) -> float:
        return self._spot

    @property
    def rate(self) -> float:
        return self._rate

    @property
    def vol(self) -> float:
        return self._vol
