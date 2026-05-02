"""
MCEngine
=========
Inspired by: Schlogl (2014) — the MCEngine (gatherer + loop).

The engine is deliberately minimal:
    for _ in range(N):
        payoffs.append(mapping(T, nb_steps))
    return PricingResult.from_payoffs(payoffs, rate, T, N, ...)

It knows NOTHING about:
  - Which process is used
  - Which payoff is computed
  - Whether variance reduction is applied

All of that is the Mapping's responsibility.
This separation means the engine never changes regardless of technique.
"""

from .mc_mapping import MCMapping
from .pricing_result import PricingResult
from sde.bs_euler_1d import BSEuler1D
from sde.bs_milstein_1d import BSMilstein1D
from sde.bs_euler_nd import BSEulerND
from sde.bs_milstein_nd import BSMilsteinND
from .basic_mapping import BasicMapping


class MCEngine:
    """
    Monte Carlo engine: loop + statistical gatherer.

    Inspired by Schlogl's MCGatherer concept.
    The engine only knows how to iterate a Mapping and collect statistics.

    Usage
    -----
    # Basic MC
    result = MCEngine().price(BasicMapping(payoff, process),
                              nb_paths=50_000, T=1.0, nb_steps=100, rate=0.05)

    # Antithetic variates — same engine, different mapping
    result = MCEngine().price(AntitheticMapping(process, payoff), ...)

    # Control variate — same engine, different mapping
    result = MCEngine().price(ControlVariateMapping(process, payoff, cv, E_cv, ...), ...)

    # QMC — same engine, BasicMapping but with SobolGenerator as process RNG
    result = MCEngine().price(BasicMapping(payoff, qmc_process), ...)
    """

    def price(
        self,
        mapping: MCMapping,
        nb_paths: int,
        T: float,
        nb_steps: int,
        rate: float,
    ) -> PricingResult:
        """
        Price an option by iterating a Mapping nb_paths times.

        Parameters
        ----------
        mapping : MCMapping
            Any callable mapping(T, nb_steps) -> float.
            Can be BasicMapping, AntitheticMapping, ControlVariateMapping, etc.
        nb_paths : int
            Number of Monte Carlo iterations (>= 2).
        T : float
            Time to maturity (> 0).
        nb_steps : int
            Number of discretization steps per path.
        rate : float
            Risk-free rate for discounting.

        Returns
        -------
        PricingResult
        """
        if nb_paths < 2:
            raise ValueError("nb_paths must be >= 2.")
        if T <= 0:
            raise ValueError("Maturity T must be > 0.")

        payoffs = [mapping(T, nb_steps) for _ in range(nb_paths)]

        return PricingResult.from_payoffs(payoffs, rate, T, nb_paths,
                                          type(mapping).__name__)

    def price_with_scheme(
        self,
        payoff,
        process,
        nb_paths: int,
        T: float,
        nb_steps: int,
        rate: float,
        scheme: str = "euler",
        multi_asset: bool = False,
    ) -> PricingResult:
        """
        Price an option using a specific integration scheme (1D or ND).

        Parameters
        ----------
        payoff : callable
            Payoff function.
        process : object
            Process object with initial value and generator.
        nb_paths : int
            Number of Monte Carlo iterations (>= 2).
        T : float
            Time to maturity (> 0).
        nb_steps : int
            Number of discretization steps per path.
        rate : float
            Risk-free rate for discounting.
        scheme : str
            Integration scheme to use ("euler" or "milstein").
        multi_asset : bool
            Whether to use multi-asset processes (ND).

        Returns
        -------
        PricingResult
        """
        if scheme == "euler":
            if multi_asset:
                process = BSEulerND(
                    initial_values=process.initial_values,
                    drift=process.drift,
                    diffusion=process.diffusion,
                    generator=process.generator,
                )
            else:
                process = BSEuler1D(
                    generator=process.generator,
                    spot=process._spot,  # Use _spot instead of initial_value
                    rate=process._rate,  # Use _rate for drift
                    vol=process._vol  # Use _vol for diffusion
                )
        elif scheme == "milstein":
            if multi_asset:
                process = BSMilsteinND(
                    initial_values=process.initial_values,
                    drift=process.drift,
                    diffusion=process.diffusion,
                    generator=process.generator,
                )
            else:
                process = BSMilstein1D(
                    initial_value=process.initial_value,
                    drift=process.drift,
                    diffusion=process.diffusion,
                    generator=process.generator,
                )
        else:
            raise ValueError(f"Unknown scheme: {scheme}")

        mapping = BasicMapping(payoff, process)
        return self.price(mapping, nb_paths, T, nb_steps, rate)

    def simulate_paths(
        self,
        process,
        nb_paths: int,
        T: float,
        nb_steps: int,
    ):
        """
        Simulate nb_paths trajectories and return all paths.

        Parameters
        ----------
        process : RandomProcess
            A configured stochastic process.
        nb_paths : int
            Number of paths to simulate.
        T : float
            Time to maturity (> 0).
        nb_steps : int
            Number of discretization steps per path.

        Returns
        -------
        list of list of SinglePath
            One entry per simulation, each entry is the list of
            SinglePath objects (one per asset dimension).
        """
        all_paths = []
        for _ in range(nb_paths):
            process.simulate(0.0, T, nb_steps)
            # Deep-copy the values so they are not overwritten on next call
            snapshot = [
                list(path.values) for path in process.get_all_paths()
            ]
            all_paths.append(snapshot)
        return all_paths
