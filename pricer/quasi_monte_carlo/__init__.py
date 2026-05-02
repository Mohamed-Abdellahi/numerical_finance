"""
pricer.quasi_monte_carlo
=========================
Quasi-Monte Carlo simulation using low-discrepancy sequences.

The only difference vs standard MC is the Mapping:
QMCMapping calls next_path() on the generator before each simulation,
ensuring each path consumes one full-dimensional Sobol point rather
than a stream of 1-D scalar draws.

Classes
-------
QMCMapping : Mapping for use with SobolPathGenerator.

Usage
-----
from random_generator.sobol import SobolPathGenerator
from random_generator.normal import NormalInverseCDF
from sde.bs_milstein_nd import BSMilsteinND
from pricer.quasi_monte_carlo import QMCMapping
from pricer.monte_carlo import MCEngine

gen     = SobolPathGenerator(dimension=n_assets * nb_steps)
normal  = NormalInverseCDF(mu=0, sigma=1, uniform=gen)
process = BSMilsteinND(normal, spots, rates, vols, corr)
result  = MCEngine().price(QMCMapping(payoff, process),
                           nb_paths=4096, T=1.0, nb_steps=52, rate=0.05)
"""

from .qmc_mapping import QMCMapping
from .qmc_antithetic_cv_mapping import QMCAntitheticCVMapping

__all__ = ["QMCMapping", "QMCAntitheticCVMapping"]