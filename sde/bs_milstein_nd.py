"""
BSMilsteinND
=============
Milstein (log-Euler exact) scheme for N correlated Black-Scholes assets.

  S_i_{t+dt} = S_i_t * exp((r_i - 0.5*sigma_i^2)*dt + sigma_i*sqrt(dt)*X_i)

where X = L*Z, Z ~ N(0, I_n), L is the Cholesky factor.

This is the exact GBM solution — no discretization error in the law.
Strong convergence order: O(dt)
"""

import math
from .black_scholes_nd import BlackScholesND


class BSMilsteinND(BlackScholesND):
    """Log-Euler exact discretization for N correlated GBM assets."""

    def simulate(self, start_time: float, end_time: float, nb_steps: int) -> None:
        self._init_paths(start_time, end_time, nb_steps)
        dt = self._paths[0].dt
        sqrt_dt = math.sqrt(dt)

        drifts = [(self._rates[i] - 0.5 * self._vols[i] ** 2) * dt
                  for i in range(self._n)]

        current = list(self._spots)
        for i in range(self._n):
            self._paths[i].insert_value(current[i])

        for _ in range(nb_steps):
            x = self._generate_correlated_normals()
            for i in range(self._n):
                dw = sqrt_dt * x[i]
                current[i] = current[i] * math.exp(drifts[i] + self._vols[i] * dw)
                self._paths[i].insert_value(current[i])
