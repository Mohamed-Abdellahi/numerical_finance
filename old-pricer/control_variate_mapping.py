"""
ControlVariateMapping
======================
Inspired by: Schlogl (2014), control_variate template decorator.

Variance reduction via static control variate:
    result = f(path) - beta * (g(path) - E[g])

where:
  f     = main payoff (e.g. arithmetic basket call)
  g     = control payoff with known analytical expectation (e.g. geometric basket)
  E[g]  = analytical expectation of g  (provided by the caller)
  beta  = Cov(f,g)/Var(g)  (estimated from a pilot simulation)

Both f and g are evaluated on the SAME simulated path — critical for the
variance reduction to work (they must be correlated).

This is a Mapping DECORATOR. It can wrap a BasicMapping or AntitheticMapping.
"""

import math
from typing import List, Optional

from payoff.payoff import Payoff
from sde.random_process import RandomProcess
from .mc_mapping import MCMapping


class ControlVariateMapping(MCMapping):
    """
    Control variate Mapping decorator.

    Inspired by Schlogl's control_variate template:
        result = f(path) - beta * (g(path) - E[g])

    Parameters
    ----------
    process : RandomProcess
        Shared stochastic process. f and g are evaluated on the same path.
    main_payoff : Payoff
        The payoff we want to price (f).
    cv_payoff : Payoff
        The control variate payoff with known analytical price (g).
        Must implement analytical_price() (e.g. GeometricBasketPayoff).
    E_cv : float
        Known analytical expectation of g, UNDISCOUNTED.
        E_cv = cv_payoff.analytical_price() * exp(r * T)
    pilot_paths : int
        Number of pilot paths used to estimate beta. Default: 2000.
    T_pilot : float
        Maturity used for the pilot simulation (same as pricing T).
    nb_steps_pilot : int
        Steps used for the pilot simulation (same as pricing nb_steps).
    """

    def __init__(
        self,
        process: RandomProcess,
        main_payoff: Payoff,
        cv_payoff: Payoff,
        E_cv: float,
        T_pilot: float,
        nb_steps_pilot: int,
        pilot_paths: int = 2000,
    ):
        self._process    = process
        self._main       = main_payoff
        self._cv         = cv_payoff
        self._E_cv       = E_cv
        self._beta       = self._estimate_beta(pilot_paths, T_pilot, nb_steps_pilot)

    def __call__(self, T: float, nb_steps: int) -> float:
        self._process.simulate(0.0, T, nb_steps)
        paths = self._process.get_all_paths()

        f = self._main(paths)
        g = self._cv(paths)
        return f - self._beta * (g - self._E_cv)

    def _estimate_beta(
        self, pilot_paths: int, T: float, nb_steps: int
    ) -> float:
        """Estimate optimal beta = Cov(f,g) / Var(g) via pilot simulation."""
        f_vals: List[float] = []
        g_vals: List[float] = []

        for _ in range(pilot_paths):
            self._process.simulate(0.0, T, nb_steps)
            paths = self._process.get_all_paths()
            f_vals.append(self._main(paths))
            g_vals.append(self._cv(paths))

        n  = pilot_paths
        mf = sum(f_vals) / n
        mg = sum(g_vals) / n
        cov   = sum((f_vals[j] - mf) * (g_vals[j] - mg) for j in range(n)) / (n - 1)
        var_g = sum((g_vals[j] - mg) ** 2 for j in range(n)) / (n - 1)
        return cov / var_g if var_g > 0 else 1.0

    @property
    def beta(self) -> float:
        """The estimated optimal control variate coefficient."""
        return self._beta
