"""
BSMilstein1D
=============
Milstein (log-Euler) discretization of the 1D Black-Scholes SDE.

For GBM with constant coefficients, Milstein = exact log-Euler:
  S_{t+dt} = S_t * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)

Strong convergence order: O(dt)  (exact for GBM)
"""

import math
from random_generator.random_generator import RandomGenerator
from .black_scholes_1d import BlackScholes1D


class BSMilstein1D(BlackScholes1D):
    """
    Milstein / log-Euler discretization of 1D GBM.

    Parameters
    ----------
    generator : RandomGenerator
    spot : float
    rate : float
    vol : float
    """

    def simulate(self, start_time: float, end_time: float, nb_steps: int) -> None:
        self._init_paths(start_time, end_time, nb_steps)
        path = self._paths[0]
        dt = path.dt
        sqrt_dt = math.sqrt(dt)
        drift = (self._rate - 0.5 * self._vol ** 2) * dt

        s = self._spot
        path.insert_value(s)

        for _ in range(nb_steps):
            z = self._generator.generate()
            s = s * math.exp(drift + self._vol * sqrt_dt * z)
            path.insert_value(s)
