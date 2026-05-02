"""
pricer — Package
================
Monte Carlo pricing library. Architecture inspired by:
Schlogl, E. — "Quantitative Finance: An Object-Oriented Approach in C++"
(Chapman & Hall, 2014).

Structure
---------
pricer/
├── pricing_result.py          — PricingResult container
├── mc_mapping.py              — MCMapping ABC (shared by all)
│
├── monte_carlo/               — Standard MC
│   ├── basic_mapping.py       — BasicMapping
│   └── mc_engine.py           — MCEngine (loop + gatherer)
│
├── variance_reduction/        — VR decorators
│   ├── antithetic_mapping.py  — AntitheticMapping
│   └── control_variate_mapping.py — ControlVariateMapping
│
├── quasi_monte_carlo/         — QMC
│   └── qmc_mapping.py         — QMCMapping (calls next_path())
│
└── american/                  — Early exercise
    └── longstaff_schwartz.py  — LongstaffSchwartz

Design principle
----------------
MCEngine is shared and unchanged regardless of technique.
Variance reduction = Mapping decorators.
QMC = swap BasicMapping for QMCMapping + SobolPathGenerator.
"""

from .pricing_result import PricingResult
from .mc_mapping import MCMapping
from .monte_carlo.basic_mapping import BasicMapping
from .monte_carlo.mc_engine import MCEngine
from .variance_reduction.antithetic_mapping import AntitheticMapping
from .variance_reduction.control_variate_mapping import ControlVariateMapping
from .quasi_monte_carlo.qmc_mapping import QMCMapping
from .american.longstaff_schwartz import LongstaffSchwartz

__all__ = [
    "PricingResult",
    "MCMapping",
    "BasicMapping",
    "MCEngine",
    "AntitheticMapping",
    "ControlVariateMapping",
    "QMCMapping",
    "LongstaffSchwartz",
]
