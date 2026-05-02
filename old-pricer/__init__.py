"""
pricer — Package
================
Monte Carlo pricing library. Architecture inspired by:
Schlogl, E. — "Quantitative Finance: An Object-Oriented Approach in C++"
(Chapman & Hall, 2014).

Design principle: the MCEngine (loop + statistics) is decoupled from
how paths are generated and payoffs are computed (MCMapping and its decorators).

Hierarchy:
  MCMapping (ABC)                      mc_mapping.py
  ├── BasicMapping                     basic_mapping.py
  ├── AntitheticMapping  (decorator)   antithetic_mapping.py
  └── ControlVariateMapping (decorator) control_variate_mapping.py

  MCEngine                             mc_engine.py
  LongstaffSchwartz                    longstaff_schwartz.py

  PricingResult                        pricing_result.py

Note: QMC is achieved by using BasicMapping with a SobolGenerator-based
process — no separate class needed. Combining techniques is user code.
"""

from .pricing_result import PricingResult
from .mc_mapping import MCMapping
from .basic_mapping import BasicMapping
from .antithetic_mapping import AntitheticMapping
from .control_variate_mapping import ControlVariateMapping
from .mc_engine import MCEngine
from .longstaff_schwartz import LongstaffSchwartz

__all__ = [
    "PricingResult",
    "MCMapping",
    "BasicMapping",
    "AntitheticMapping",
    "ControlVariateMapping",
    "MCEngine",
    "LongstaffSchwartz",
]
