from sde.random_process import RandomProcess
import numpy as np

class HestonModel(RandomProcess):
    def __init__(self, generator, v0, kappa, theta, sigma, rho):
        """
        Heston model for stochastic volatility.
        v_t = kappa * (theta - v_t) * dt + sigma * sqrt(v_t) * dW_t
        """
        super().__init__(generator)
        self.v0 = v0  # Initial variance
        self.kappa = kappa  # Mean reversion speed
        self.theta = theta  # Long-term variance
        self.sigma = sigma  # Volatility of volatility
        self.rho = rho  # Correlation with asset price

    def simulate(self, start_time, end_time, nb_steps):
        dt = (end_time - start_time) / nb_steps
        times = np.linspace(start_time, end_time, nb_steps + 1)
        v = np.zeros(nb_steps + 1)
        v[0] = self.v0

        for i in range(nb_steps):
            z1 = self.generator.generate()
            z2 = self.generator.generate()
            w1 = z1
            w2 = self.rho * z1 + np.sqrt(1 - self.rho**2) * z2

            v[i + 1] = v[i] + self.kappa * (self.theta - v[i]) * dt + self.sigma * np.sqrt(max(v[i], 0)) * np.sqrt(dt) * w1

        self.path = v
        self.times = times