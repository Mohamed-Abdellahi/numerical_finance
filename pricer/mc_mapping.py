"""
MCMapping — Abstract Base Class
=================================
Inspired by: Schlogl, E. — "Quantitative Finance: An Object-Oriented
Approach in C++" (Chapman & Hall, 2014).

A Mapping is a functor:
    mapping(T, nb_steps) -> float   (one payoff realization)

The MCEngine is just a loop over this functor:
    payoffs = [mapping(T, nb_steps) for _ in range(N)]

Variance reduction = Mapping decorators.
QMC             = QMCMapping (calls next_path() before each simulation).
All share the same MCEngine — it never changes.
"""

from abc import ABC, abstractmethod


class MCMapping(ABC):
    """
    Abstract base for all Monte Carlo path-to-payoff mappings.

    A mapping encapsulates everything for ONE Monte Carlo draw:
      - generate a path (simulate the process)
      - compute the payoff on that path

    Concrete subclasses
    -------------------
    BasicMapping             standard MC
    QMCMapping               quasi-random MC (calls next_path())
    AntitheticMapping        variance reduction decorator
    ControlVariateMapping    variance reduction decorator
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
