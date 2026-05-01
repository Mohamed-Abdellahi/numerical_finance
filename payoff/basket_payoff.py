"""
Basket Payoffs
==============
Multi-asset basket call and put payoffs with arbitrary weights.

  BasketCallPayoff:  max( Σ αi·Si_T - K , 0 )
  BasketPutPayoff:   max( K - Σ αi·Si_T , 0 )

Weights αi can be negative (short positions allowed per project spec).
When n=1 and α=[1.0], this reduces exactly to the vanilla payoff.
"""

from typing import List
from sde.single_path import SinglePath
from .payoff import Payoff


class BasketCallPayoff(Payoff):
    """
    European Basket Call payoff.

    Payoff = max( Σ αi·Si_T - K , 0 )

    Parameters
    ----------
    weights : list of float
        Weights [α1, ..., αn]. Can be negative (project allows it).
    strike : float
        Strike price K.
    """

    def __init__(self, weights: List[float], strike: float):
        super().__init__(strike)
        if not weights:
            raise ValueError("Weights list must not be empty.")
        self._weights = list(weights)

    def __call__(self, paths: List[SinglePath]) -> float:
        """
        Parameters
        ----------
        paths : list of SinglePath
            One path per asset, len(paths) must match len(weights).

        Returns
        -------
        float
            max( Σ αi·Si_T - K , 0 )
        """
        if len(paths) != len(self._weights):
            raise ValueError(
                f"Expected {len(self._weights)} paths, got {len(paths)}."
            )
        basket_value = sum(
            self._weights[i] * paths[i].values[-1]
            for i in range(len(self._weights))
        )
        return max(basket_value - self._strike, 0.0)

    def basket_value_at_step(self, paths: List[SinglePath], step: int) -> float:
        """
        Compute the weighted basket value at a given time step index.
        Used by Longstaff-Schwartz in the pricer.

        Parameters
        ----------
        paths : list of SinglePath
        step : int
            Index in the path values list.

        Returns
        -------
        float
            Σ αi·Si_{t_step}
        """
        return sum(
            self._weights[i] * paths[i].values[step]
            for i in range(len(self._weights))
        )

    @property
    def weights(self) -> List[float]:
        return list(self._weights)

    @property
    def n_assets(self) -> int:
        return len(self._weights)


class BasketPutPayoff(Payoff):
    """
    European Basket Put payoff.

    Payoff = max( K - Σ αi·Si_T , 0 )

    Parameters
    ----------
    weights : list of float
        Weights [α1, ..., αn].
    strike : float
        Strike price K.
    """

    def __init__(self, weights: List[float], strike: float):
        super().__init__(strike)
        if not weights:
            raise ValueError("Weights list must not be empty.")
        self._weights = list(weights)

    def __call__(self, paths: List[SinglePath]) -> float:
        """
        Returns
        -------
        float
            max( K - Σ αi·Si_T , 0 )
        """
        if len(paths) != len(self._weights):
            raise ValueError(
                f"Expected {len(self._weights)} paths, got {len(paths)}."
            )
        basket_value = sum(
            self._weights[i] * paths[i].values[-1]
            for i in range(len(self._weights))
        )
        return max(self._strike - basket_value, 0.0)

    def basket_value_at_step(self, paths: List[SinglePath], step: int) -> float:
        """Compute the weighted basket value at a given time step index."""
        return sum(
            self._weights[i] * paths[i].values[step]
            for i in range(len(self._weights))
        )

    @property
    def weights(self) -> List[float]:
        return list(self._weights)

    @property
    def n_assets(self) -> int:
        return len(self._weights)
