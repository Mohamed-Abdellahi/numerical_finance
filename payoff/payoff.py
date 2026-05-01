"""
Payoff — Abstract Base Class
==============================
Every payoff is a callable: given a list of simulated paths,
it returns the undiscounted payoff amount.

Design choice:
  __call__(paths: List[SinglePath]) -> float
  
  - paths[i] is the SinglePath for asset i
  - The payoff looks at the terminal value (or all values for path-dependent)
  - Returns the raw payoff (before discounting)
  - Discounting is handled by the pricer, not the payoff
"""

from abc import ABC, abstractmethod
from typing import List
from sde.single_path import SinglePath


class Payoff(ABC):
    """
    Abstract base class for all option payoffs.

    A Payoff is a callable that maps a list of asset paths
    to a non-negative payoff amount (before discounting).

    Parameters
    ----------
    strike : float
        Strike price K (>= 0).
    """

    def __init__(self, strike: float):
        if strike < 0:
            raise ValueError("Strike must be non-negative.")
        self._strike = strike

    @abstractmethod
    def __call__(self, paths: List[SinglePath]) -> float:
        """
        Compute the undiscounted payoff for one set of simulated paths.

        Parameters
        ----------
        paths : list of SinglePath
            Simulated paths, one per asset dimension.

        Returns
        -------
        float
            The undiscounted payoff (>= 0).
        """
        ...

    @property
    def strike(self) -> float:
        return self._strike
