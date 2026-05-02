"""
pricer.american
================
Early exercise pricing via the Longstaff-Schwartz algorithm.

Completely separate from the MCEngine/Mapping architecture.
Longstaff-Schwartz requires storing all paths and running
backward dynamic programming.

Classes
-------
LongstaffSchwartz   : Bermudan basket option pricer (basic + transparent QMC).
LongstaffSchwartzVR : Extended pricer with antithetic, CV, and full VR.
"""

from .longstaff_schwartz    import LongstaffSchwartz
from .longstaff_schwartz_vr import LongstaffSchwartzVR

__all__ = ["LongstaffSchwartz", "LongstaffSchwartzVR"]
