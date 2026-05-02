"""
LongstaffSchwartz
==================
Pricer for Bermudan options using the Longstaff-Schwartz algorithm.

Algorithm
---------
1. Simulate M paths forward - store all asset values at all time steps.
2. Initialize: everyone exercises at maturity T.
3. Backward induction from step N-1 to 1:
   a. Intrinsic value at step k: g = max(basket - K, 0)
   b. For ITM paths: regress discounted cashflow on polynomial basis
   c. Exercise now if intrinsic >= estimated continuation value
4. Price = mean over paths of exp(-r*tau)*cashflow(tau)

The class exposes protected helpers (_simulate_paths, _run_ls, _find_sobol)
that LongstaffSchwartzVR reuses to add variance reduction.

QMC compatibility: if the process generator implements next_path()
(SobolPathGenerator), it is called automatically before each simulation.

Reference: Longstaff & Schwartz (2001).
"""

import numpy as np
from typing import List, Tuple

from payoff.bermudan_payoff import BermudanBasketPayoff
from sde.random_process import RandomProcess
from pricer.pricing_result import PricingResult


def _polynomial_basis(x: np.ndarray, degree: int) -> np.ndarray:
    """Build design matrix [1, x, x^2, ..., x^degree], shape (n, degree+1)."""
    X = np.ones((len(x), degree + 1))
    for d in range(1, degree + 1):
        X[:, d] = x ** d
    return X


class LongstaffSchwartz:
    """
    Longstaff-Schwartz algorithm for Bermudan basket option pricing.

    Parameters
    ----------
    regression_degree : int
        Degree of the polynomial basis for continuation value regression.
        Default: 2 (basis: 1, S, S^2).
    """

    def __init__(self, regression_degree: int = 2):
        if regression_degree < 1:
            raise ValueError("regression_degree must be >= 1.")
        self._degree = regression_degree

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def price(
        self,
        bermudan_payoff: BermudanBasketPayoff,
        process: RandomProcess,
        nb_paths: int,
        nb_steps: int,
        rate: float,
    ) -> PricingResult:
        """
        Price a Bermudan Basket option.

        Works transparently with both pseudo-random (L'Ecuyer) and
        quasi-random (Sobol) processes - the generator's next_path()
        is called automatically when present.

        Parameters
        ----------
        bermudan_payoff : BermudanBasketPayoff
        process : RandomProcess  (BSMilsteinND recommended)
        nb_paths : int
        nb_steps : int
        rate : float

        Returns
        -------
        PricingResult
        """
        T, _, n_assets, K, weights, dt, exercise_steps = \
            self._parse_payoff(bermudan_payoff, nb_steps)

        all_values = self._simulate_paths(process, nb_paths, n_assets, T, nb_steps)
        cashflow, tau_step = self._run_ls(
            all_values, nb_paths, exercise_steps, weights, K, dt, rate
        )

        time_to_exercise = tau_step * dt
        discounted = np.exp(-rate * time_to_exercise) * cashflow

        return PricingResult.from_payoffs(
            discounted.tolist(), rate=0.0, T=T,
            nb_paths=nb_paths, method="Longstaff-Schwartz"
        )

    # ------------------------------------------------------------------
    # Protected helpers reused by LongstaffSchwartzVR
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_payoff(bermudan_payoff: BermudanBasketPayoff, nb_steps: int):
        """Extract payoff parameters and map exercise dates to step indices."""
        T             = bermudan_payoff.maturity
        ex_dates      = bermudan_payoff.exercise_dates
        n_assets      = bermudan_payoff.n_assets
        K             = bermudan_payoff.strike
        weights       = bermudan_payoff.weights
        dt            = T / nb_steps
        exercise_steps = [round(t / dt) for t in ex_dates]
        return T, ex_dates, n_assets, K, weights, dt, exercise_steps

    @staticmethod
    def _find_sobol(process: RandomProcess):
        """
        Walk the generator chain to find a SobolPathGenerator (next_path()).
        Returns the Sobol generator if found, else None.
        Handles both direct use and use via NormalInverseCDF wrapper.
        """
        gen = process._generator
        if hasattr(gen, "next_path"):
            return gen
        if hasattr(gen, "_uniform") and hasattr(gen._uniform, "next_path"):
            return gen._uniform
        return None

    def _simulate_paths(
        self,
        process: RandomProcess,
        nb_paths: int,
        n_assets: int,
        T: float,
        nb_steps: int,
    ) -> np.ndarray:
        """
        Simulate nb_paths paths and store them in a
        (n_assets, nb_paths, nb_steps+1) array.
        Calls next_path() automatically for QMC generators.
        """
        sobol      = self._find_sobol(process)
        all_values = np.zeros((n_assets, nb_paths, nb_steps + 1))

        for j in range(nb_paths):
            if sobol is not None:
                sobol.next_path()
            process.simulate(0.0, T, nb_steps)
            paths = process.get_all_paths()
            for i in range(n_assets):
                all_values[i, j, :] = paths[i].values

        return all_values

    def _run_ls(
        self,
        all_values: np.ndarray,
        total_paths: int,
        exercise_steps: List[int],
        weights: List[float],
        K: float,
        dt: float,
        rate: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Longstaff-Schwartz backward induction on a pre-simulated path matrix.

        Parameters
        ----------
        all_values   : np.ndarray, shape (n_assets, total_paths, nb_steps+1)
        total_paths  : int
        exercise_steps, weights, K, dt, rate : as usual

        Returns
        -------
        cashflow  : np.ndarray, shape (total_paths,)  - undiscounted payoff at tau
        tau_step  : np.ndarray, shape (total_paths,)  - optimal exercise step index
        """
        n_assets = all_values.shape[0]

        # Initialise: everyone exercises at maturity
        tau_step = np.full(total_paths, exercise_steps[-1], dtype=int)
        basket_T = sum(
            weights[i] * all_values[i, :, exercise_steps[-1]]
            for i in range(n_assets)
        )
        cashflow = np.maximum(basket_T - K, 0.0)

        # Backward induction
        for k in reversed(range(len(exercise_steps) - 1)):
            step_k  = exercise_steps[k]
            step_dt = (tau_step - step_k) * dt

            basket_k    = sum(
                weights[i] * all_values[i, :, step_k]
                for i in range(n_assets)
            )
            intrinsic_k = np.maximum(basket_k - K, 0.0)

            itm_mask = intrinsic_k > 0
            if not np.any(itm_mask):
                continue

            disc_cashflow = np.exp(-rate * step_dt[itm_mask]) * cashflow[itm_mask]
            x_itm         = basket_k[itm_mask]
            X             = _polynomial_basis(x_itm, self._degree)

            alpha, _, _, _ = np.linalg.lstsq(X, disc_cashflow, rcond=None)
            continuation   = X @ alpha

            exercise_now = intrinsic_k[itm_mask] >= continuation
            itm_indices  = np.where(itm_mask)[0]
            ex_idx       = itm_indices[exercise_now]

            tau_step[ex_idx] = step_k
            cashflow[ex_idx] = intrinsic_k[ex_idx]

        return cashflow, tau_step
