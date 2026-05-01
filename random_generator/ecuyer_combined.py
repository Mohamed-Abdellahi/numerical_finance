"""
EcuyerCombined
==============
Combined generator from L'Ecuyer (1988).
Combines TWO Linear Congruential Generators for a much longer period
and better statistical properties.

Algorithm (exp):
  X1_{n+1} = (40014 * X1_n) mod 2147483563
  X2_{n+1} = (40692 * X2_n) mod 2147483399
  Z_n       = (X1_n - X2_n) mod 2147483562
  U_n       = Z_n / 2147483563       (or (Z_n + 1) / 2147483564 to avoid 0)


"""

from .pseudo_generator import PseudoGenerator
from .linear_congruential import LinearCongruential


class EcuyerCombined(PseudoGenerator):
    """
    L'Ecuyer's combined generator — combines two LCGs for a period of ~2^61.

    Parameters
    ----------
    seed1 : int
        Seed for the first LCG. Default: 1.
    seed2 : int
        Seed for the second LCG. Default: 1.
    """


    _M1 = 2_147_483_563
    _M2 = 2_147_483_399
    _A1 = 40_014
    _A2 = 40_692

    def __init__(self, seed1: int = 1, seed2: int = 1):
        # We use seed1 as the "main" seed for PseudoGenerator
        super().__init__(seed=seed1)
        self._first = LinearCongruential(
            multiplier=self._A1,
            increment=0,
            modulus=self._M1,
            seed=seed1,
        )
        self._second = LinearCongruential(
            multiplier=self._A2,
            increment=0,
            modulus=self._M2,
            seed=seed2,
        )

    def generate(self) -> float:
        """
        Advance both LCGs, combine and return a U(0,1) value.

        Z = (X1 - X2) mod (M1 - 1)
        U = Z / M1   (mapped to (0, 1) )
        """
       # print(f"[DEBUG] Before generate: x1={self._first.current}, x2={self._second.current}")

        x1 = self._first.current
        x2 = self._second.current

        # Advance both generators
        self._first.generate()
        self._second.generate()

        x1_new = self._first.current
        x2_new = self._second.current

       # print(f"[DEBUG] After generate: x1_new={x1_new}, x2_new={x2_new}")

        z = (x1_new - x2_new) % (self._M1 - 1)
        # Ensure z is never 0
        if z == 0:
            z = self._M1 - 1

        return z / self._M1

    def reset(self):
        """Reset both sub-generators to their initial seeds."""
        self._first.reset()
        self._second.reset()
