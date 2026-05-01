"""
BSMilstein2D
=============
Milstein (log-Euler) scheme for 2 correlated Black-Scholes assets.

Correlated increments via:
  dW1 = sqrt(dt) * Z1
  dW2 = sqrt(dt) * (rho*Z1 + sqrt(1-rho^2)*Z2)
  Z1, Z2 ~ N(0,1) independent
"""

import math
from .black_scholes_2d import BlackScholes2D


class BSMilstein2D(BlackScholes2D):
    """
    Log-Euler discretization of 2 correlated GBM assets.

    S1_{t+dt} = S1_t * exp((r1 - 0.5*sigma1^2)*dt + sigma1*dW1)
    S2_{t+dt} = S2_t * exp((r2 - 0.5*sigma2^2)*dt + sigma2*dW2)
    """

    def simulate(self, start_time: float, end_time: float, nb_steps: int) -> None:
        self._init_paths(start_time, end_time, nb_steps)
        dt = self._paths[0].dt
        sqrt_dt = math.sqrt(dt)

        drift1 = (self._rate1 - 0.5 * self._vol1 ** 2) * dt
        drift2 = (self._rate2 - 0.5 * self._vol2 ** 2) * dt

        s1, s2 = self._spot1, self._spot2
        self._paths[0].insert_value(s1)
        self._paths[1].insert_value(s2)

        for _ in range(nb_steps):
            z1 = self._generator.generate()
            z2 = self._generator.generate()

            dw1 = sqrt_dt * z1
            dw2 = sqrt_dt * (self._rho * z1 + self._sqrt_one_minus_rho2 * z2)

            s1 = s1 * math.exp(drift1 + self._vol1 * dw1)
            s2 = s2 * math.exp(drift2 + self._vol2 * dw2)

            self._paths[0].insert_value(s1)
            self._paths[1].insert_value(s2)
