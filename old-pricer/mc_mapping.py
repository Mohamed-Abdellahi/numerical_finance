"""
MCMapping — Abstract Base Class
=================================
Inspired by: Schlogl, E. — "Quantitative Finance: An Object-Oriented
Approach in C++" (Chapman & Hall, 2014), Chapter on Monte Carlo simulation.

In Schlogl's design, a Mapping is a functor:
    mapping(T, nb_steps) -> float (one discounted payoff realization)

The MCEngine (pricer) is just a loop:
    payoffs = [mapping(T, nb_steps) for _ in range(N)]

Variance reduction = Mapping decorators, not separate pricers.
This lets each quant work on their own Mapping independently.
"""

from abc import ABC, abstractmethod


class MCMapping(ABC):
    """
    Abstract base for Monte Carlo path-to-payoff mappings.

    A mapping encapsulates everything that happens for ONE Monte Carlo draw:
      - generate a path (simulate the process)
      - compute the payoff on that path

    Variance reduction techniques are implemented as Mapping decorators.
    The MCEngine stays the same regardless of which mapping is used.
    """

    @abstractmethod
    def __call__(self, T: float, nb_steps: int) -> float:
        """
        Simulate one path and return one (possibly variance-reduced) payoff.

        Parameters
        ----------
        T : float
            Time to maturity.
        nb_steps : int
            Number of discretization steps.

        Returns
        -------
        float
            One realized payoff value (NOT yet discounted by the engine).
        """
        ...
