"""
SinglePath
==========
Represents one simulated path of a stochastic process over [StartTime, EndTime].

Stores all intermediate values at each time step.
Provides:
  - InsertValue(val)          ← add a new value to the path
  - GetState(time) → float    ← interpolate/retrieve value at a given time
  - GetAllValues() → list     ← return the full path

"""

from typing import List


class SinglePath:
    """
    Stores one simulated path of a stochastic process.

    Parameters
    ----------
    start_time : float
        Start of the time interval.
    end_time : float
        End of the time interval.
    nb_steps : int
        Number of time steps in the discretization.
    """

    def __init__(self, start_time: float, end_time: float, nb_steps: int):
        if end_time <= start_time:
            raise ValueError("end_time must be strictly greater than start_time.")
        if nb_steps < 1:
            raise ValueError("nb_steps must be >= 1.")

        self._start_time: float = start_time
        self._end_time: float = end_time
        self._nb_steps: int = nb_steps
        self._values: List[float] = []

        # Step size dt
        self._dt: float = (end_time - start_time) / nb_steps

    def insert_value(self, val: float) -> None:
        """Append a new value to the path."""
        self._values.append(val)

    def get_state(self, time: float) -> float:
        """
        Retrieve the path value at a given time by linear interpolation
        between the two closest stored values.

        Parameters
        ----------
        time : float
            The time at which to retrieve the path value.

        Returns
        -------
        float
            The path value at the requested time.
        """
        if not self._values:
            raise RuntimeError("Path is empty. Call Simulate() first.")
        if time < self._start_time or time > self._end_time:
            raise ValueError(
                f"time={time} is outside [{self._start_time}, {self._end_time}]."
            )

        # Compute the index of the nearest time step
        idx = (time - self._start_time) / self._dt
        lower = int(idx)
        upper = lower + 1

        # Clamp to valid range
        lower = max(0, min(lower, len(self._values) - 1))
        upper = max(0, min(upper, len(self._values) - 1))

        if lower == upper:
            return self._values[lower]

        # Linear interpolation
        frac = idx - lower
        return self._values[lower] * (1 - frac) + self._values[upper] * frac

    def get_all_values(self) -> List[float]:
        """Return a copy of all stored path values."""
        return list(self._values)

    def reset(self) -> None:
        """Clear all stored values (to re-simulate)."""
        self._values = []

    def __len__(self) -> int:
        return len(self._values)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def start_time(self) -> float:
        return self._start_time

    @property
    def end_time(self) -> float:
        return self._end_time

    @property
    def nb_steps(self) -> int:
        return self._nb_steps

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def values(self) -> List[float]:
        return self._values
