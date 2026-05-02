"""
QMCMapping
===========
Mapping for Quasi-Monte Carlo simulation with SobolPathGenerator.

The only difference vs BasicMapping is the call to next_path() before
each simulation. This is essential for correct QMC behaviour:

  BasicMapping  : calls generator.generate() N*T times sequentially
                  → uses N*T consecutive 1-D Sobol scalars
                  → WRONG: loses low-discrepancy structure

  QMCMapping    : calls generator.next_path() once per simulation
                  → draws one (n_assets * nb_steps)-dimensional Sobol point
                  → CORRECT: each simulation is one point in the full space

The MCEngine stays unchanged — it just calls mapping(T, nb_steps) in a loop.

Requirements
------------
The process generator must implement next_path() — i.e. it must be a
SobolPathGenerator. If not, QMCMapping raises a clear RuntimeError.
"""

from payoff.payoff import Payoff
from sde.random_process import RandomProcess
from pricer.mc_mapping import MCMapping


class QMCMapping(MCMapping):
    """
    Quasi-Monte Carlo mapping.

    Like BasicMapping, but calls generator.next_path() before each
    simulation to draw a new full-dimensional Sobol point.

    Parameters
    ----------
    payoff : Payoff
        Any payoff object (BasketCallPayoff, ...).
    process : RandomProcess
        A process whose generator is a SobolPathGenerator
        (must implement next_path()).
    """

# Dans qmc_mapping.py, remplace __init__ et __call__ par :

    def __init__(self, payoff: Payoff, process: RandomProcess):
        # next_path() peut être sur le générateur direct OU sur son uniforme sous-jacent
        gen = process._generator
        if hasattr(gen, "next_path"):
            self._sobol = gen
        elif hasattr(gen, "_uniform") and hasattr(gen._uniform, "next_path"):
            self._sobol = gen._uniform
        else:
            raise TypeError(
                "QMCMapping requires a generator with next_path(). "
                "Use SobolPathGenerator as the process generator."
            )
        self._payoff  = payoff
        self._process = process

    def __call__(self, T: float, nb_steps: int) -> float:
        self._sobol.next_path()
        self._process.simulate(0.0, T, nb_steps)
        return self._payoff(self._process.get_all_paths())

    @property
    def payoff(self) -> Payoff:
        return self._payoff

    @property
    def process(self) -> RandomProcess:
        return self._process
