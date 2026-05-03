"""
LongstaffSchwartzVR
====================
Variance-reduction extensions of the Longstaff-Schwartz pricer.

Three techniques, applied cumulatively:
  1. QMC (transparent): pass a process with a SobolPathGenerator.
  2. Antithetic variates: price_antithetic()
  3. Static control variate: price_with_cv()
  4. All combined: price_full_vr()

Extends LongstaffSchwartz and reuses its protected helpers.
"""

import math
import numpy as np
from typing import List, Tuple

from payoff.bermudan_payoff import BermudanBasketPayoff
from payoff.geometric_basket_payoff import GeometricBasketPayoff
from sde.random_process import RandomProcess
from pricer.pricing_result import PricingResult
from pricer.american.longstaff_schwartz import LongstaffSchwartz


class LongstaffSchwartzVR(LongstaffSchwartz):
    """
    Longstaff-Schwartz with variance reduction for Bermudan basket options.

    Parameters
    ----------
    regression_degree : int
        Polynomial degree for the continuation value regression. Default: 2.
    """

    def price_antithetic(
        self,
        bermudan_payoff: BermudanBasketPayoff,
        process: RandomProcess,
        nb_paths: int,
        nb_steps: int,
        rate: float,
    ) -> PricingResult:
        """
        LS with antithetic variates.

        For each simulated Z-path, the antithetic -Z path is derived analytically
        from the log-returns. LS runs on the combined 2M paths; payoffs are then
        averaged in (Z, -Z) pairs.

        Parameters
        ----------
        bermudan_payoff : BermudanBasketPayoff
        process : RandomProcess
        nb_paths : int
        nb_steps : int
        rate : float
        """
        T, _, n_assets, K, weights, dt, exercise_steps = \
            self._parse_payoff(bermudan_payoff, nb_steps)

        rates_list, vols_list = self._get_rates_vols(process, n_assets)

        z_values   = self._simulate_paths(process, nb_paths, n_assets, T, nb_steps)
        anti_values = self._antithetic_paths(z_values, n_assets, nb_steps, rates_list, vols_list, dt)

        combined    = np.concatenate([z_values, anti_values], axis=1)
        total_paths = 2 * nb_paths

        cashflow, tau_step = self._run_ls(
            combined, total_paths, exercise_steps, weights, K, dt, rate
        )

        discounted = np.exp(-rate * tau_step * dt) * cashflow
        paired = 0.5 * (discounted[:nb_paths] + discounted[nb_paths:])

        return PricingResult.from_payoffs(
            paired.tolist(), rate=0.0, T=T,
            nb_paths=nb_paths, method="Longstaff-Schwartz (Antithetic)"
        )

    def price_with_cv(
        self,
        bermudan_payoff: BermudanBasketPayoff,
        process: RandomProcess,
        nb_paths: int,
        nb_steps: int,
        rate: float,
        cv_payoff: GeometricBasketPayoff,
        E_cv: float,
        pilot_paths: int = 2000,
    ) -> PricingResult:
        """
        LS with a static control variate (European geometric basket call).

        Applies f_j - beta*(g_j - E[g]) after the LS backward induction,
        where beta is estimated from a pilot simulation.

        Parameters
        ----------
        bermudan_payoff : BermudanBasketPayoff
        process : RandomProcess
        nb_paths : int
        nb_steps : int
        rate : float
        cv_payoff : GeometricBasketPayoff
        E_cv : float
            Undiscounted E[g] = cv_payoff.analytical_price() * exp(r*T).
        pilot_paths : int
            Paths used to estimate beta. Default: 2000.
        """
        T, _, n_assets, K, weights, dt, exercise_steps = \
            self._parse_payoff(bermudan_payoff, nb_steps)

        E_g_disc = cv_payoff.analytical_price()

        beta = self._estimate_beta_ls(
            process, pilot_paths, n_assets, T, nb_steps,
            exercise_steps, weights, K, dt, rate,
            cv_payoff, E_g_disc
        )

        all_values         = self._simulate_paths(process, nb_paths, n_assets, T, nb_steps)
        cashflow, tau_step = self._run_ls(all_values, nb_paths, exercise_steps, weights, K, dt, rate)

        f = np.exp(-rate * tau_step * dt) * cashflow
        g = self._geo_basket_disc(all_values, cv_payoff, n_assets, rate, T, K)

        cv_payoffs = f - beta * (g - E_g_disc)

        return PricingResult.from_payoffs(
            cv_payoffs.tolist(), rate=0.0, T=T,
            nb_paths=nb_paths, method="Longstaff-Schwartz (CV)"
        )

    def price_full_vr(
        self,
        bermudan_payoff: BermudanBasketPayoff,
        process: RandomProcess,
        nb_paths: int,
        nb_steps: int,
        rate: float,
        cv_payoff: GeometricBasketPayoff,
        E_cv: float,
        pilot_paths: int = 2000,
    ) -> PricingResult:
        """
        LS combining QMC + antithetic variates + control variate.

        Parameters
        ----------
        bermudan_payoff : BermudanBasketPayoff
        process : RandomProcess
            Should use a SobolPathGenerator.
        nb_paths : int
        nb_steps : int
        rate : float
        cv_payoff : GeometricBasketPayoff
        E_cv : float
            E_cv = cv_payoff.analytical_price() * exp(r*T).
        pilot_paths : int
        """
        T, _, n_assets, K, weights, dt, exercise_steps = \
            self._parse_payoff(bermudan_payoff, nb_steps)

        rates_list, vols_list = self._get_rates_vols(process, n_assets)
        E_g_disc = cv_payoff.analytical_price()

        beta = self._estimate_beta_ls_anti(
            process, pilot_paths, n_assets, T, nb_steps,
            exercise_steps, weights, K, dt, rate,
            cv_payoff, E_g_disc, rates_list, vols_list
        )

        z_values    = self._simulate_paths(process, nb_paths, n_assets, T, nb_steps)
        anti_values = self._antithetic_paths(z_values, n_assets, nb_steps, rates_list, vols_list, dt)

        combined    = np.concatenate([z_values, anti_values], axis=1)
        total_paths = 2 * nb_paths
        cashflow, tau_step = self._run_ls(
            combined, total_paths, exercise_steps, weights, K, dt, rate
        )

        f = np.exp(-rate * tau_step * dt) * cashflow
        g = self._geo_basket_disc(combined, cv_payoff, n_assets, rate, T, K)

        cv_all  = f - beta * (g - E_g_disc)
        paired  = 0.5 * (cv_all[:nb_paths] + cv_all[nb_paths:])

        return PricingResult.from_payoffs(
            paired.tolist(), rate=0.0, T=T,
            nb_paths=nb_paths, method="Longstaff-Schwartz (QMC+Anti+CV)"
        )

    @staticmethod
    def _antithetic_paths(
        all_values: np.ndarray,
        n_assets: int,
        nb_steps: int,
        rates: List[float],
        vols: List[float],
        dt: float,
    ) -> np.ndarray:
        """
        Build antithetic paths from original paths via log-return reflection.

        log(S_anti_{t+1}/S_anti_t)_i = 2*(r_i - sigma_i^2/2)*dt - log(S_{t+1}/S_t)_i
        """
        anti = np.zeros_like(all_values)
        for i in range(n_assets):
            anti[i, :, 0] = all_values[i, :, 0]
            drift_i = (rates[i] - 0.5 * vols[i] ** 2) * dt
            for t in range(nb_steps):
                S_t   = np.maximum(all_values[i, :, t],   1e-12)
                S_tp1 = np.maximum(all_values[i, :, t+1], 1e-12)
                log_ret_anti = 2.0 * drift_i - np.log(S_tp1 / S_t)
                anti[i, :, t+1] = anti[i, :, t] * np.exp(log_ret_anti)
        return anti

    @staticmethod
    def _geo_basket_disc(
        all_values: np.ndarray,
        cv_payoff: GeometricBasketPayoff,
        n_assets: int,
        rate: float,
        T: float,
        K: float,
    ) -> np.ndarray:
        """Discounted European geometric basket payoff at maturity for each path."""
        norm_w  = cv_payoff._norm_w
        S_T     = [np.maximum(all_values[i, :, -1], 1e-12) for i in range(n_assets)]
        log_geo = sum(norm_w[i] * np.log(S_T[i]) for i in range(n_assets))
        G_T     = np.exp(log_geo)
        return math.exp(-rate * T) * np.maximum(G_T - K, 0.0)

    def _estimate_beta_ls(
        self,
        process, pilot_paths, n_assets, T, nb_steps,
        exercise_steps, weights, K, dt, rate,
        cv_payoff, E_g_disc,
    ) -> float:
        """Estimate beta = Cov(f, g) / Var(g) from a pilot LS simulation."""
        av = self._simulate_paths(process, pilot_paths, n_assets, T, nb_steps)
        cf, tau = self._run_ls(av, pilot_paths, exercise_steps, weights, K, dt, rate)
        f_vals = np.exp(-rate * tau * dt) * cf
        g_vals = self._geo_basket_disc(av, cv_payoff, n_assets, rate, T, K)
        return self._beta_from_samples(f_vals, g_vals)

    def _estimate_beta_ls_anti(
        self,
        process, pilot_paths, n_assets, T, nb_steps,
        exercise_steps, weights, K, dt, rate,
        cv_payoff, E_g_disc, rates_list, vols_list,
    ) -> float:
        """Estimate beta using antithetic pilot paths, consistent with price_full_vr."""
        z_av     = self._simulate_paths(process, pilot_paths, n_assets, T, nb_steps)
        av_anti  = self._antithetic_paths(z_av, n_assets, nb_steps, rates_list, vols_list, dt)
        combined = np.concatenate([z_av, av_anti], axis=1)

        cf, tau  = self._run_ls(combined, 2 * pilot_paths, exercise_steps, weights, K, dt, rate)
        f_all    = np.exp(-rate * tau * dt) * cf
        g_all    = self._geo_basket_disc(combined, cv_payoff, n_assets, rate, T, K)

        f_paired = 0.5 * (f_all[:pilot_paths] + f_all[pilot_paths:])
        g_paired = 0.5 * (g_all[:pilot_paths] + g_all[pilot_paths:])
        return self._beta_from_samples(f_paired, g_paired)

    @staticmethod
    def _beta_from_samples(f: np.ndarray, g: np.ndarray) -> float:
        """beta = Cov(f, g) / Var(g)."""
        n  = len(f)
        mf, mg = f.mean(), g.mean()
        cov   = float(np.sum((f - mf) * (g - mg))) / (n - 1)
        var_g = float(np.sum((g - mg) ** 2)) / (n - 1)
        return cov / var_g if var_g > 0 else 1.0

    @staticmethod
    def _get_rates_vols(process: RandomProcess, n_assets: int):
        """Extract per-asset rates and vols from the process."""
        if hasattr(process, "_rates"):
            return list(process._rates), list(process._vols)
        return [process._rate] * n_assets, [process._vol] * n_assets
