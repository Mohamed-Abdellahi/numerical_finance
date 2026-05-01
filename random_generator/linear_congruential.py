"""
LinearCongruential
==================
Linear Congruential Generator (LCG): the classic pseudo-random generator.

Recurrence:    X_{n+1} = (a * X_n + c) mod m
Uniform value: U_n = X_n / m

Parameters match the example here
  a = 40014, c = 0, m = 2147483563


"""

from .pseudo_generator import PseudoGenerator


class LinearCongruential(PseudoGenerator):
    """
    Linear Congruential Generator.

    Parameters
    ----------
    multiplier : int
        Multiplier (a) in the recurrence.
    increment : int
        Increment (c). Set to 0 for a multiplicative generator.
    modulus : int
        Modulus (m). Must be strictly positive.
    seed : int
        Initial seed. Must satisfy 0 < seed < modulus.
    """

    def __init__(
        self,
        multiplier: int = 40014,
        increment: int = 0,
        modulus: int = 2_147_483_563,
        seed: int = 1,
    ):
        if modulus <= 0:
            raise ValueError("Modulus must be strictly positive.")
        if not (0 < seed < modulus):
            raise ValueError(f"Seed must satisfy 0 < seed < modulus ({modulus}).")

        super().__init__(seed=seed)
        self._multiplier = multiplier
        self._increment = increment
        self._modulus = modulus

    def generate(self) -> float:
        """
        Advance the congruential recurrence and return U in (0, 1).

        X_{n+1} = (a * X_n + c) mod m
        U        = X_{n+1} / m
        """
        self._current = (self._multiplier * self._current + self._increment) % self._modulus
        return self._current / self._modulus

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def multiplier(self) -> int:
        return self._multiplier

    @property
    def increment(self) -> int:
        return self._increment

    @property
    def modulus(self) -> int:
        return self._modulus
