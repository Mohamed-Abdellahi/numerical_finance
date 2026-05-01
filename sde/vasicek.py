from sde.random_process import RandomProcess
import numpy as np

class VasicekModel(RandomProcess):
    def __init__(self, generator, r0, kappa, theta, sigma):
        """
        Vasicek model for interest rates.
        dr_t = kappa * (theta - r_t) * dt + sigma * dW_t
        """
        super().__init__(generator)
        self.r0 = r0  # Initial rate
        self.kappa = kappa  # Mean reversion speed
        self.theta = theta  # Long-term mean
        self.sigma = sigma  # Volatility

    def simulate(self, start_time, end_time, nb_steps):
        dt = (end_time - start_time) / nb_steps
        times = np.linspace(start_time, end_time, nb_steps + 1)
        r = np.zeros(nb_steps + 1)
        r[0] = self.r0

        for i in range(nb_steps):
            z = self.generator.generate()
            r[i + 1] = r[i] + self.kappa * (self.theta - r[i]) * dt + self.sigma * np.sqrt(dt) * z

        self.path = r
        self.times = times