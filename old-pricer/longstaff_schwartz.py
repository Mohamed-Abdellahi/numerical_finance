"""
LongstaffSchwartz
==================
Pricer for Bermudan options using the Longstaff-Schwartz algorithm.

Completely separate from MonteCarloPricer — this is a fundamentally
different algorithm (backward dynamic programming, not simple averaging).

Algorithm:
  1. Simulate M paths forward — store ALL asset values at ALL time steps
  2. Initialize: τ_j = T (everyone exercises at maturity)
  3. Backward induction from step N-1 to 1:
     a. Intrinsic value: g_j = max(basket_j - K, 0)
     b. For ITM paths: regress e^{-r*(τ_j-t_k)} * cashflow_j on basis(basket_j)
     c. Exercise if g_j >= estimated continuation value
  4. Price = mean over j of  e^{-r*τ_j} * cashflow_j

Reference: Longstaff & Schwartz (2001), "Valuing American Options by Simulation"
"""

import numpy as np
from typing import List

from payoff.bermudan_payoff import BermudanBasketPayoff
from sde.random_process import RandomProcess
from .pricing_result import PricingResult


def _polynomial_basis(x: np.ndarray, degree: int) -> np.ndarray:
    """
    Build the polynomial design matrix [1, x, x^2, ..., x^degree].

    Parameters
    ----------
    x : np.ndarray, shape (n,)
    degree : int

    Returns
    -------
    np.ndarray, shape (n, degree+1)
    """
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
        Degree of the polynomial used to estimate the continuation value.
        Default: 2 (basis: 1, S, S²).
    """

    def __init__(self, regression_degree: int = 2):
        if regression_degree < 1:
            raise ValueError("regression_degree must be >= 1.")
        self._degree = regression_degree

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

        Parameters
        ----------
        bermudan_payoff : BermudanBasketPayoff
            Defines weights, strike, and exercise dates.
        process : RandomProcess
            N-asset Black-Scholes process with a shared generator.
        nb_paths : int
            Number of simulated paths M.
        nb_steps : int
            Number of time steps N. Exercise dates are mapped to steps.
        rate : float
            Risk-free rate r.

        Returns
        -------
        PricingResult
        """
        T        = bermudan_payoff.maturity
        ex_dates = bermudan_payoff.exercise_dates
        n_assets = bermudan_payoff.n_assets
        K        = bermudan_payoff.strike
        weights  = bermudan_payoff.weights
        dt       = T / nb_steps

        # Map exercise dates → step indices
        exercise_steps = [round(t / dt) for t in ex_dates]

        # ---------------------------------------------------------------
        # Step 1: Simulate and store all M paths
        # all_values[asset, path, step]
        # ---------------------------------------------------------------
        all_values = np.zeros((n_assets, nb_paths, nb_steps + 1))
        for j in range(nb_paths):
            process.simulate(0.0, T, nb_steps)
            paths = process.get_all_paths()
            for i in range(n_assets):
                all_values[i, j, :] = paths[i].values

        # ---------------------------------------------------------------
        # Step 2: Initialize — exercise at maturity
        # ---------------------------------------------------------------
        tau_step = np.full(nb_paths, exercise_steps[-1], dtype=int)

        basket_T = sum(weights[i] * all_values[i, :, exercise_steps[-1]]
                       for i in range(n_assets))
        cashflow = np.maximum(basket_T - K, 0.0)

        # ---------------------------------------------------------------
        # Step 3: Backward induction
        # ---------------------------------------------------------------
        for k in reversed(range(len(exercise_steps) - 1)):
            step_k  = exercise_steps[k]
            step_dt = (tau_step - step_k) * dt

            basket_k    = sum(weights[i] * all_values[i, :, step_k]
                              for i in range(n_assets))
            intrinsic_k = np.maximum(basket_k - K, 0.0)

            itm_mask = intrinsic_k > 0
            if not np.any(itm_mask):
                continue

            disc_cashflow = np.exp(-rate * step_dt[itm_mask]) * cashflow[itm_mask]
            x_itm = basket_k[itm_mask]
            X     = _polynomial_basis(x_itm, self._degree)

            alpha, _, _, _ = np.linalg.lstsq(X, disc_cashflow, rcond=None)
            continuation   = X @ alpha

            exercise_now   = intrinsic_k[itm_mask] >= continuation
            itm_indices    = np.where(itm_mask)[0]
            ex_idx         = itm_indices[exercise_now]

            tau_step[ex_idx] = step_k
            cashflow[ex_idx] = intrinsic_k[ex_idx]

        # ---------------------------------------------------------------
        # Step 4: Discount and average
        # ---------------------------------------------------------------
        time_to_exercise = tau_step * dt
        discounted       = np.exp(-rate * time_to_exercise) * cashflow

        # Discounting already applied per-path, so rate=0 in from_payoffs
        return PricingResult.from_payoffs(
            discounted.tolist(), rate=0.0, T=T,
            nb_paths=nb_paths, method="Longstaff-Schwartz"
        )
