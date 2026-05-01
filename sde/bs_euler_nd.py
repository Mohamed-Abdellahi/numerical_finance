"""
BSEulerND
==========
Euler-Maruyama discretization for N correlated Black-Scholes assets.

  S_i_{t+dt} = S_i_t * (1 + r_i*dt + sigma_i*sqrt(dt)*X_i)

where X = L*Z, Z ~ N(0, I_n), L is the Cholesky factor.

Strong convergence order: O(sqrt(dt))
"""

import math
from .black_scholes_nd import BlackScholesND


class BSEulerND(BlackScholesND):
    """Euler-Maruyama for N correlated GBM assets."""

    def simulate(self, start_time: float, end_time: float, nb_steps: int) -> None:
        self._init_paths(start_time, end_time, nb_steps)
        dt = self._paths[0].dt
        sqrt_dt = math.sqrt(dt)

        current = list(self._spots)
        for i in range(self._n):
            self._paths[i].insert_value(current[i])

        for _ in range(nb_steps):
            x = self._generate_correlated_normals()
            for i in range(self._n):
                dw = sqrt_dt * x[i]
                current[i] = current[i] + self._rates[i] * current[i] * dt \
                             + self._vols[i] * current[i] * dw
                current[i] = max(current[i], 0.0)
                self._paths[i].insert_value(current[i])
