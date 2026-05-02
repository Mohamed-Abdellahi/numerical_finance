"""
BasicMapping
=============
Inspired by: Schlogl (2014) — the simplest Mapping functor.

Wraps a stochastic process and a payoff into a single callable:
    mapping(T, nb_steps) -> payoff(simulated_path)

This is the building block for all other mappings (decorators).
"""

from payoff.payoff import Payoff
from sde.random_process import RandomProcess
from .mc_mapping import MCMapping


class BasicMapping(MCMapping):
    """
    Standard Monte Carlo mapping: simulate one path, return one payoff.

    Parameters
    ----------
    payoff : Payoff
        Any payoff object (VanillaCallPayoff, BasketCallPayoff, ...).
    process : RandomProcess
        A configured stochastic process with a persistent generator.
    """

    def __init__(self, payoff: Payoff, process: RandomProcess):
        self._payoff  = payoff
        self._process = process

    def __call__(self, T: float, nb_steps: int) -> float:
        self._process.simulate(0.0, T, nb_steps)
        return self._payoff(self._process.get_all_paths())

    @property
    def payoff(self) -> Payoff:
        return self._payoff

    @property
    def process(self) -> RandomProcess:
        return self._process
