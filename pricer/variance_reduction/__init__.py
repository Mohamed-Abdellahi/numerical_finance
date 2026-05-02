"""
pricer.variance_reduction
==========================
Variance reduction decorators.

These are Mapping decorators — they wrap any MCMapping and reduce
its estimator variance without touching the MCEngine.

Classes
-------
AntitheticMapping      : antithetic variates (returns 0.5*(f(Z) + f(-Z)))
ControlVariateMapping  : static control variate (f - beta*(g - E[g]))
"""

from .antithetic_mapping import AntitheticMapping
from .control_variate_mapping import ControlVariateMapping

__all__ = ["AntitheticMapping", "ControlVariateMapping"]