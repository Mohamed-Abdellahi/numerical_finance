"""
BSEuler1D
==========
Euler-Maruyama discretization of the 1D Black-Scholes SDE.

Scheme:
  S_{t+dt} = S_t * (1 + r*dt + sigma*sqrt(dt)*Z),   Z ~ N(0,1)

Strong convergence order: O(sqrt(dt))
"""

import math
from random_generator.random_generator import RandomGenerator
from .black_scholes_1d import BlackScholes1D


class BSEuler1D(BlackScholes1D):
    """
    Euler-Maruyama discretization of 1D GBM.

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

        s = self._spot
        path.insert_value(s)

        for _ in range(nb_steps):
            z = self._generator.generate()
            s = s + self._rate * s * dt + self._vol * s * sqrt_dt * z
            s = max(s, 0.0)
            path.insert_value(s)
