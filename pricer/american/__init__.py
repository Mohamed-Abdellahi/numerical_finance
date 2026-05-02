"""
pricer.american
================
Early exercise pricing via the Longstaff-Schwartz algorithm.

Completely separate from the MCEngine/Mapping architecture —
Longstaff-Schwartz requires storing all paths and running
backward dynamic programming, which doesn't fit the
one-path-at-a-time Mapping pattern.

Classes
-------
LongstaffSchwartz : Bermudan basket option pricer.
"""

from .longstaff_schwartz import LongstaffSchwartz

__all__ = ["LongstaffSchwartz"]