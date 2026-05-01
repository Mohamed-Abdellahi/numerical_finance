"""
AntitheticMapping
==================
Inspired by: Schlogl (2014), antithetic_sampling template decorator.

Variance reduction via antithetic variates:
    For each Gaussian draw Z, also evaluate the path with -Z.
    Return 0.5 * (payoff(Z) + payoff(-Z)).

This is a Mapping DECORATOR: it wraps a process + payoff and produces
a variance-reduced payoff. The MCEngine stays unchanged.

Variance reduction factor: 1 + Corr(f(Z), f(-Z)) < 1 for monotone payoffs.
"""

from payoff.payoff import Payoff
from sde.random_process import RandomProcess
from sde.bs_milstein_1d import BSMilstein1D
from sde.bs_milstein_nd import BSMilsteinND
from sde.cholesky import recover_correlation
from random_generator.normal import NormalBoxMuller
from .mc_mapping import MCMapping


class _AntitheticNormal(NormalBoxMuller):
    """
    Internal adapter: returns -Z for each Z drawn from the paired generator.
    Shares the same random stream as the original, producing the mirror path.
    Not exported — implementation detail of AntitheticMapping.
    """
    def __init__(self, paired: NormalBoxMuller):
        self._paired = paired  # do NOT call super().__init__()

    def generate(self) -> float:
        return -self._paired.generate()


class AntitheticMapping(MCMapping):
    """
    Antithetic variates Mapping decorator.

    Inspired by Schlogl's antithetic_sampling template:
        result = 0.5 * (base_mapping(Z) + base_mapping(-Z))

    Parameters
    ----------
    process : BSMilstein1D or BSMilsteinND
        The original stochastic process.
    payoff : Payoff
        The payoff to evaluate on each path.
    """

    def __init__(self, process: RandomProcess, payoff: Payoff):
        if not isinstance(process, (BSMilstein1D, BSMilsteinND)):
            raise TypeError("AntitheticMapping requires BSMilstein1D or BSMilsteinND.")
        self._process      = process
        self._payoff       = payoff
        self._anti_process = self._build_antithetic(process)

    def __call__(self, T: float, nb_steps: int) -> float:
        self._process.simulate(0.0, T, nb_steps)
        p1 = self._payoff(self._process.get_all_paths())

        self._anti_process.simulate(0.0, T, nb_steps)
        p2 = self._payoff(self._anti_process.get_all_paths())

        return 0.5 * (p1 + p2)

    def _build_antithetic(self, process: RandomProcess) -> RandomProcess:
        """Build the mirror process using -Z for every Gaussian draw."""
        anti_gen = _AntitheticNormal(process._generator)
        if isinstance(process, BSMilstein1D):
            return BSMilstein1D(anti_gen, process._spot, process._rate, process._vol)
        else:
            corr = recover_correlation(process._cholesky)
            return BSMilsteinND(anti_gen, process._spots, process._rates,
                                process._vols, corr)
