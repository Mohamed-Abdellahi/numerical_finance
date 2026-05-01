"""
Bermudan Basket Payoff
=======================
Defines the structure of a Bermudan option on a basket of assets.

A Bermudan option can be exercised at a discrete set of dates:
  t0 = 0 < t1 < t2 < ... < tN = T

At each exercise date tk, if the holder exercises, they receive:
  max( Σ αi·Si_tk - K , 0 )

The BermudanBasketPayoff class stores:
  - The exercise dates
  - The weights and strike
  - The intrinsic value function (payoff if exercised at step k)

The Longstaff-Schwartz pricing algorithm is implemented in the pricer
(MonteCarloPricer.price_bermudan), not here. This class just defines
what gets paid and when it can be paid.

Design separation:
  BermudanBasketPayoff  →  defines the contract (what + when)
  MonteCarloPricer      →  implements the LS algorithm (how to price it)
"""

from typing import List
from sde.single_path import SinglePath
from .payoff import Payoff


class BermudanBasketPayoff(Payoff):
    """
    Bermudan Basket Call option payoff structure.

    Parameters
    ----------
    weights : list of float
        Weights [α1, ..., αn] for the basket. Can be negative.
    strike : float
        Strike price K.
    exercise_dates : list of float
        Sorted list of exercise dates [t0, t1, ..., tN] with t0=0, tN=T.
        Must be increasing.
    """

    def __init__(
        self,
        weights: List[float],
        strike: float,
        exercise_dates: List[float],
    ):
        super().__init__(strike)

        if not weights:
            raise ValueError("Weights list must not be empty.")
        if len(exercise_dates) < 2:
            raise ValueError("At least 2 exercise dates required (t0 and tN).")
        if any(exercise_dates[i] >= exercise_dates[i + 1]
               for i in range(len(exercise_dates) - 1)):
            raise ValueError("Exercise dates must be strictly increasing.")

        self._weights = list(weights)
        self._exercise_dates = list(exercise_dates)

    def __call__(self, paths: List[SinglePath]) -> float:
        """
        Terminal payoff (at maturity, if never exercised early).
        Returns max( Σ αi·Si_T - K , 0 ).

        Note: for Bermudan pricing, use MonteCarloPricer.price_bermudan()
        which implements the full Longstaff-Schwartz algorithm.
        """
        if len(paths) != len(self._weights):
            raise ValueError(
                f"Expected {len(self._weights)} paths, got {len(paths)}."
            )
        basket_T = sum(
            self._weights[i] * paths[i].values[-1]
            for i in range(len(self._weights))
        )
        return max(basket_T - self._strike, 0.0)

    def intrinsic_value(self, paths: List[SinglePath], step: int) -> float:
        """
        Compute the intrinsic value (payoff if exercised now) at path step k.

        Payoff = max( Σ αi·Si_{step} - K , 0 )

        Parameters
        ----------
        paths : list of SinglePath
            Simulated paths (one per asset).
        step : int
            Index into path.values[] corresponding to the exercise date.

        Returns
        -------
        float
            Intrinsic value at this step.
        """
        basket = sum(
            self._weights[i] * paths[i].values[step]
            for i in range(len(self._weights))
        )
        return max(basket - self._strike, 0.0)

    def basket_value_at_step(self, paths: List[SinglePath], step: int) -> float:
        """
        Compute the raw weighted basket value at a given step (before applying payoff).
        Used by Longstaff-Schwartz as the regression feature.

        Parameters
        ----------
        paths : list of SinglePath
        step : int

        Returns
        -------
        float
            Σ αi·Si_{step}
        """
        return sum(
            self._weights[i] * paths[i].values[step]
            for i in range(len(self._weights))
        )

    @property
    def weights(self) -> List[float]:
        return list(self._weights)

    @property
    def exercise_dates(self) -> List[float]:
        return list(self._exercise_dates)

    @property
    def n_assets(self) -> int:
        return len(self._weights)

    @property
    def maturity(self) -> float:
        return self._exercise_dates[-1]

    @property
    def n_exercise_dates(self) -> int:
        return len(self._exercise_dates)
