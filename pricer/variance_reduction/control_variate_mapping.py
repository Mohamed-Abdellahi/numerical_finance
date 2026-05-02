"""
ControlVariateMapping
======================
Inspired by: Schlogl (2014), control_variate template decorator.

Variance reduction via static control variate:
    result = f(path) - beta * (g(path) - E[g])

where:
  f     = main payoff (e.g. arithmetic basket call)
  g     = control payoff with known analytical expectation (e.g. geometric basket)
  E[g]  = analytical expectation of g (undiscounted, provided by the caller)
  beta  = Cov(f,g) / Var(g)  (estimated from a pilot simulation)

Both f and g are evaluated on the SAME simulated path.
"""

from typing import List

from payoff.payoff import Payoff
from sde.random_process import RandomProcess
from pricer.mc_mapping import MCMapping


class ControlVariateMapping(MCMapping):
    """
    Control variate Mapping decorator.

        result = f(path) - beta * (g(path) - E[g])

    Parameters
    ----------
    process : RandomProcess
        Shared stochastic process. f and g are evaluated on the same path.
    main_payoff : Payoff
        The payoff we want to price (f).
    cv_payoff : Payoff
        The control variate payoff with known analytical price (g).
    E_cv : float
        Known analytical expectation of g, UNDISCOUNTED.
        E_cv = analytical_price * exp(r * T)
    T_pilot : float
        Maturity used for the pilot simulation.
    nb_steps_pilot : int
        Steps used for the pilot simulation.
    pilot_paths : int
        Number of pilot paths to estimate beta. Default: 2000.
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
        self._process = process
        self._main    = main_payoff
        self._cv      = cv_payoff
        self._E_cv    = E_cv
        self._beta    = self._estimate_beta(pilot_paths, T_pilot, nb_steps_pilot)

    def __call__(self, T: float, nb_steps: int) -> float:
        gen = self._process._generator
        if hasattr(gen, "next_path"):
            gen.next_path()
        elif hasattr(gen, "_uniform") and hasattr(gen._uniform, "next_path"):
            gen._uniform.next_path()
        self._process.simulate(0.0, T, nb_steps)
        paths = self._process.get_all_paths()
        f = self._main(paths)
        g = self._cv(paths)
        return f - self._beta * (g - self._E_cv)

    def _estimate_beta(self, pilot_paths: int, T: float, nb_steps: int) -> float:
        """Estimate optimal beta = Cov(f,g) / Var(g) via pilot simulation."""
        f_vals: List[float] = []
        g_vals: List[float] = []

        # Detect if the generator needs next_path() (QMC case)
        gen = self._process._generator
        sobol = None
        if hasattr(gen, "next_path"):
            sobol = gen
        elif hasattr(gen, "_uniform") and hasattr(gen._uniform, "next_path"):
            sobol = gen._uniform

        for _ in range(pilot_paths):
            if sobol is not None:
                sobol.next_path()
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
