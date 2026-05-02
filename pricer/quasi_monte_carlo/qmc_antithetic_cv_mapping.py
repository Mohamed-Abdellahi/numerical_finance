"""
QMCAntitheticCVMapping
=======================
Combines three variance reduction techniques in one clean mapping:

  1. QMC (Sobol)         : low-discrepancy sequence via SobolPathGenerator
  2. Antithetic variates : mirror path using 1-u instead of u
                           (since Phi^{-1}(1-u) = -Phi^{-1}(u) by symmetry)
  3. Control variate     : geometric basket as static control

For each simulation, the estimator is:

  result = 0.5 * ( (f(Z)  - beta*(g(Z)  - E[g]))
                 + (f(-Z) - beta*(g(-Z) - E[g])) )

where:
  f     = arithmetic basket call (main payoff)
  g     = geometric basket call  (control variate, known analytical price)
  Z     = correlated Gaussian vector from Sobol point u
  -Z    = antithetic path from 1-u  (Phi^{-1}(1-u) = -Phi^{-1}(u))
  beta  = Cov(f,g) / Var(g), estimated from pilot simulation
  E[g]  = analytical expectation of g (undiscounted)

Architecture
------------
- _AntitheticSobolAdapter : reads the same Sobol vector as the original
                            process but returns 1-u for each coordinate.
                            Has its own index — does not interfere with
                            the original SobolPathGenerator index.
- QMCAntitheticCVMapping  : orchestrates next_path(), two simulations,
                            and the CV correction.
"""

from typing import List, Optional

from payoff.payoff import Payoff
from sde.random_process import RandomProcess
from sde.bs_milstein_1d import BSMilstein1D
from sde.bs_milstein_nd import BSMilsteinND
from sde.cholesky import recover_correlation
from random_generator.uniform_generator import UniformGenerator
from random_generator.normal import NormalInverseCDF
from random_generator.sobol import SobolPathGenerator
from pricer.mc_mapping import MCMapping


class _AntitheticSobolAdapter(UniformGenerator):
    """
    Antithetic adapter for SobolPathGenerator.

    Reads the same current Sobol vector as the original generator
    but returns (1 - u) for each coordinate instead of u.

    Since NormalInverseCDF computes Phi^{-1}(u), returning 1-u gives
    Phi^{-1}(1-u) = -Phi^{-1}(u), which is exactly the antithetic draw.

    Has its own index — does not advance the original SobolPathGenerator.
    Must call prepare() after next_path() on the original, before simulating
    the antithetic path.

    Parameters
    ----------
    sobol : SobolPathGenerator
        The original Sobol generator whose current vector to mirror.
    """

    def __init__(self, sobol: SobolPathGenerator):
        super().__init__()
        self._sobol = sobol
        self._index = 0

    def prepare(self) -> None:
        """
        Reset internal index to 0.
        Must be called after the original process has finished simulating,
        before the antithetic process starts.
        """
        self._index = 0

    def generate(self) -> float:
        """
        Return 1 - u for the current coordinate of the Sobol vector.
        Phi^{-1}(1 - u) = -Phi^{-1}(u) by symmetry of the normal distribution.
        """
        if self._index >= len(self._sobol._current_vector):
            raise RuntimeError(
                "_AntitheticSobolAdapter: index out of range. "
                "Make sure prepare() is called before each antithetic simulation."
            )
        u = self._sobol._current_vector[self._index]
        self._index += 1
        return 1.0 - u

    # next_path() is not needed — the adapter reads from the original vector.


class QMCAntitheticCVMapping(MCMapping):
    """
    QMC + Antithetic Variates + Static Control Variate mapping.

    Combines all three variance reduction techniques in a single callable.
    For each call:
      1. next_path()  : draw one full-dimensional Sobol point
      2. simulate Z   : original path → f(Z), g(Z)
      3. simulate -Z  : antithetic path (via 1-u) → f(-Z), g(-Z)
      4. return 0.5 * ( CV_estimator(Z) + CV_estimator(-Z) )

    Parameters
    ----------
    process : BSMilstein1D or BSMilsteinND
        Process whose generator is NormalInverseCDF(SobolPathGenerator).
    main_payoff : Payoff
        The payoff to price (f).
    cv_payoff : Payoff
        Control variate payoff with known analytical price (g).
    E_cv : float
        Undiscounted analytical expectation of g.
        E_cv = cv_payoff.analytical_price() * exp(r * T)
    T_pilot : float
        Maturity for the pilot simulation.
    nb_steps_pilot : int
        Number of steps for the pilot simulation.
    pilot_paths : int
        Number of pilot paths to estimate beta. Default: 2000.
    """

    def __init__(
        self,
        process: RandomProcess,
        main_payoff: Payoff,
        cv_payoff: Payoff,
        E_cv: float,
        T_pilot: float,
        nb_steps_pilot: int,
        pilot_paths: int = 2000,
    ):
        if not isinstance(process, (BSMilstein1D, BSMilsteinND)):
            raise TypeError(
                "QMCAntitheticCVMapping requires BSMilstein1D or BSMilsteinND."
            )

        # Retrieve the SobolPathGenerator from the process generator chain
        self._sobol = self._find_sobol(process)
        if self._sobol is None:
            raise TypeError(
                "QMCAntitheticCVMapping requires a SobolPathGenerator "
                "in the process generator chain."
            )

        self._process = process
        self._main    = main_payoff
        self._cv      = cv_payoff
        self._E_cv    = E_cv

        # Build the antithetic process
        self._anti_adapter  = _AntitheticSobolAdapter(self._sobol)
        self._anti_process  = self._build_antithetic_process(process)

        # Estimate beta from pilot simulation
        self._beta = self._estimate_beta(pilot_paths, T_pilot, nb_steps_pilot)

    # ------------------------------------------------------------------
    # MCMapping interface
    # ------------------------------------------------------------------

    def __call__(self, T: float, nb_steps: int) -> float:
        # Step 1: draw one Sobol point
        self._sobol.next_path()

        # Step 2: simulate original path Z
        self._process.simulate(0.0, T, nb_steps)
        paths_z = self._process.get_all_paths()
        f_z = self._main(paths_z)
        g_z = self._cv(paths_z)

        # Step 3: simulate antithetic path -Z (via 1-u)
        self._anti_adapter.prepare()
        self._anti_process.simulate(0.0, T, nb_steps)
        paths_mz = self._anti_process.get_all_paths()
        f_mz = self._main(paths_mz)
        g_mz = self._cv(paths_mz)

        # Step 4: combine CV estimators
        cv_z  = f_z  - self._beta * (g_z  - self._E_cv)
        cv_mz = f_mz - self._beta * (g_mz - self._E_cv)
        return 0.5 * (cv_z + cv_mz)

    # ------------------------------------------------------------------
    # Pilot simulation to estimate beta = Cov(f,g) / Var(g)
    # Uses both Z and -Z paths (consistent with the full estimator)
    # ------------------------------------------------------------------

    def _estimate_beta(self, pilot_paths: int, T: float, nb_steps: int) -> float:
        """Estimate beta = Cov(f,g) / Var(g) using antithetic pilot paths."""
        f_vals: List[float] = []
        g_vals: List[float] = []

        for _ in range(pilot_paths):
            self._sobol.next_path()

            # Z path
            self._process.simulate(0.0, T, nb_steps)
            paths_z = self._process.get_all_paths()
            f_z = self._main(paths_z)
            g_z = self._cv(paths_z)

            # -Z path
            self._anti_adapter.prepare()
            self._anti_process.simulate(0.0, T, nb_steps)
            paths_mz = self._anti_process.get_all_paths()
            f_mz = self._main(paths_mz)
            g_mz = self._cv(paths_mz)

            # Average over the pair (consistent with __call__)
            f_vals.append(0.5 * (f_z  + f_mz))
            g_vals.append(0.5 * (g_z  + g_mz))

        n  = pilot_paths
        mf = sum(f_vals) / n
        mg = sum(g_vals) / n
        cov   = sum((f_vals[j] - mf) * (g_vals[j] - mg) for j in range(n)) / (n - 1)
        var_g = sum((g_vals[j] - mg) ** 2 for j in range(n)) / (n - 1)
        return cov / var_g if var_g > 0 else 1.0

    # ------------------------------------------------------------------
    # Build the antithetic process — same parameters, adapter as generator
    # ------------------------------------------------------------------

    def _build_antithetic_process(self, process: RandomProcess) -> RandomProcess:
        """
        Build the mirror process using _AntitheticSobolAdapter as the
        uniform source for a new NormalInverseCDF generator.
        """
        anti_normal = NormalInverseCDF(
            mu=0.0, sigma=1.0, uniform=self._anti_adapter
        )
        if isinstance(process, BSMilstein1D):
            return BSMilstein1D(
                anti_normal, process._spot, process._rate, process._vol
            )
        else:
            corr = recover_correlation(process._cholesky)
            return BSMilsteinND(
                anti_normal, process._spots, process._rates,
                process._vols, corr
            )

    # ------------------------------------------------------------------
    # Helper: find SobolPathGenerator in the generator chain
    # ------------------------------------------------------------------

    @staticmethod
    def _find_sobol(process: RandomProcess) -> Optional[SobolPathGenerator]:
        """
        Walk the generator chain to find the SobolPathGenerator.
        Handles both direct use and use via NormalInverseCDF wrapper.
        """
        gen = process._generator
        if isinstance(gen, SobolPathGenerator):
            return gen
        if hasattr(gen, "_uniform") and isinstance(gen._uniform, SobolPathGenerator):
            return gen._uniform
        return None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def beta(self) -> float:
        """The estimated optimal control variate coefficient."""
        return self._beta