"""
HeadTail Generator
==================
Simulates a fair coin flip: returns 0 or 1 with equal probability.

This is the simplest discrete generator and is also used internally
by the NormalRejectionSampling generator (to assign a random sign).

Reference: Generator Architecture PDF (Kaiza Amouh)
"""

from .discrete_generator import DiscreteGenerator
from .uniform_generator import UniformGenerator


class HeadTail(DiscreteGenerator):
    """
    Fair coin flip generator.
    Returns 0 (tails) or 1 (heads) with equal probability 0.5.

    Parameters
    ----------
    uniform : UniformGenerator
        Underlying uniform generator.
    """

    def __init__(self, uniform: UniformGenerator):
        self.__uniform = uniform
        print(f"[DEBUG] HeadTail initialized with uniform generator: {self.__uniform}")
        super().__init__(target_mean=0.5, target_variance=0.25)

    def generate(self) -> float:
        """Return 0 or 1 with equal probability."""
        if self.__uniform is None:
            raise AttributeError("[DEBUG] __uniform is None in HeadTail.generate(). Ensure it is initialized correctly.")
        return 1.0 if self.__uniform.generate() <= 0.5 else 0.0
