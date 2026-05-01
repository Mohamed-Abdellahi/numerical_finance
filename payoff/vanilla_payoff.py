"""
Vanilla Payoffs
================
Standard single-asset European call and put payoffs.

  VanillaCallPayoff:  max(S_T - K, 0)
  VanillaPutPayoff:   max(K - S_T, 0)

Expects exactly one path in the paths list.
The terminal value is the last stored value in the path.
"""

from typing import List
from sde.single_path import SinglePath
from .payoff import Payoff


class VanillaCallPayoff(Payoff):
    """
    European Call payoff on a single asset.

    Payoff = max(S_T - K, 0)

    Parameters
    ----------
    strike : float
        Strike price K.
    """

    def __call__(self, paths: List[SinglePath]) -> float:
        """
        Parameters
        ----------
        paths : list of SinglePath
            Must contain exactly one path (single asset).

        Returns
        -------
        float
            max(S_T - K, 0)
        """
        if len(paths) != 1:
            raise ValueError(
                f"VanillaCallPayoff expects 1 path, got {len(paths)}."
            )
        s_T = paths[0].values[-1]
        return max(s_T - self._strike, 0.0)


class VanillaPutPayoff(Payoff):
    """
    European Put payoff on a single asset.

    Payoff = max(K - S_T, 0)

    Parameters
    ----------
    strike : float
        Strike price K.
    """

    def __call__(self, paths: List[SinglePath]) -> float:
        """
        Parameters
        ----------
        paths : list of SinglePath
            Must contain exactly one path (single asset).

        Returns
        -------
        float
            max(K - S_T, 0)
        """
        if len(paths) != 1:
            raise ValueError(
                f"VanillaPutPayoff expects 1 path, got {len(paths)}."
            )
        s_T = paths[0].values[-1]
        return max(self._strike - s_T, 0.0)
