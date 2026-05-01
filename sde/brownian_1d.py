"""
Brownian1D
===========
1-dimensional standard Brownian motion (Wiener process).

Simulation:
  B_0 = 0
  B_{t+dt} = B_t + sqrt(dt) * Z,   Z ~ N(0,1)
"""

import math
from random_generator.random_generator import RandomGenerator
from .random_process import RandomProcess


class Brownian1D(RandomProcess):
    """
    1-dimensional standard Brownian motion.

    Parameters
    ----------
    generator : RandomGenerator
        Must produce N(0,1) samples (e.g., NormalBoxMuller).
    """

    def __init__(self, generator: RandomGenerator):
        super().__init__(generator, dimension=1)

    def simulate(self, start_time: float, end_time: float, nb_steps: int) -> None:
        """
        Simulate the Brownian path over [start_time, end_time].

        Parameters
        ----------
        start_time : float
        end_time : float
        nb_steps : int
        """
        self._init_paths(start_time, end_time, nb_steps)
        path = self._paths[0]
        dt = path.dt

        current = 0.0
        path.insert_value(current)

        for _ in range(nb_steps):
            current += math.sqrt(dt) * self._generator.generate()
            path.insert_value(current)
