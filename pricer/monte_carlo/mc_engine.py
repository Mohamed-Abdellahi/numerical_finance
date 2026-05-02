"""
MCEngine
=========
Inspired by: Schlogl (2014) — the MCEngine (gatherer + loop).

The engine is deliberately minimal:
    for _ in range(N):
        payoffs.append(mapping(T, nb_steps))
    return PricingResult.from_payoffs(payoffs, ...)

It knows NOTHING about:
  - Which process is used        (pseudo-random or quasi-random)
  - Which payoff is computed     (vanilla, basket, barrier ...)
  - Whether variance reduction is applied

All of that is the Mapping's responsibility.
This separation means the engine never changes regardless of technique.

Compatible with: BasicMapping, QMCMapping, AntitheticMapping,
                 ControlVariateMapping, and any future decorator.
"""

from pricer.mc_mapping import MCMapping
from pricer.pricing_result import PricingResult


class MCEngine:
    """
    Monte Carlo engine: loop + statistical gatherer.

    The engine is the same for MC, QMC, and all variance reduction
    combinations. Only the Mapping changes.

    Usage
    -----
    # Standard MC
    engine = MCEngine()
    result = engine.price(BasicMapping(payoff, process),
                          nb_paths=50_000, T=1.0, nb_steps=52, rate=0.05)

    # QMC — same engine, different mapping
    result = engine.price(QMCMapping(payoff, qmc_process), ...)

    # QMC + antithetic — same engine, composed mappings
    result = engine.price(AntitheticMapping(QMCMapping(payoff, process)), ...)
    """

    def price(
        self,
        mapping: MCMapping,
        nb_paths: int,
        T: float,
        nb_steps: int,
        rate: float,
    ) -> PricingResult:
        """
        Price an option by iterating a Mapping nb_paths times.

        Parameters
        ----------
        mapping : MCMapping
            Any callable mapping(T, nb_steps) -> float.
        nb_paths : int
            Number of Monte Carlo iterations (>= 2).
        T : float
            Time to maturity (> 0).
        nb_steps : int
            Number of discretization steps per path.
        rate : float
            Risk-free rate for discounting.

        Returns
        -------
        PricingResult
        """
        if nb_paths < 2:
            raise ValueError("nb_paths must be >= 2.")
        if T <= 0:
            raise ValueError("Maturity T must be > 0.")
        if nb_steps < 1:
            raise ValueError("nb_steps must be >= 1.")

        payoffs = [mapping(T, nb_steps) for _ in range(nb_paths)]

        return PricingResult.from_payoffs(
            payoffs, rate, T, nb_paths, method=type(mapping).__name__
        )
