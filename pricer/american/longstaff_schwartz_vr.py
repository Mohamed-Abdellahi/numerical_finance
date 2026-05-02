"""
LongstaffSchwartzVR
====================
Variance-reduction extensions of the Longstaff-Schwartz algorithm.

Three techniques are layered cumulatively, following the same progression
as Question 1 (b):

  1. QMC  (transparent)    : pass a process whose generator is a
                             SobolPathGenerator — the parent's
                             _simulate_paths() calls next_path() automatically.

  2. Antithetic variates   : price_antithetic()
                             For each simulated path (Z), compute the
                             antithetic path (-Z) analytically from the
                             log-returns, without touching the RNG.
                             LS is run on the combined 2M paths for a
                             better regression; payoffs are then averaged
                             in (Z, -Z) pairs so PricingResult.from_payoffs
                             correctly captures the negative correlation.

  3. Static control variate: price_with_cv()
                             After running LS, apply the correction
                                 f_j - beta*(g_j - E[g])
                             where g = European geometric basket call at T
                             (closed-form analytical price), evaluated on
                             the same simulated path.
                             beta = Cov(f, g)/Var(g) estimated from a pilot.

  4. All combined          : price_full_vr()
                             QMC process + antithetic paths + CV correction.

Architecture note
-----------------
LongstaffSchwartzVR extends LongstaffSchwartz and reuses its protected
helpers (_simulate_paths, _run_ls, _find_sobol, _parse_payoff).
The public price() method from the parent is unchanged.
"""

import math
import numpy as np
from typing import List, Optional, Tuple

from payoff.bermudan_payoff import BermudanBasketPayoff
from payoff.geometric_basket_payoff import GeometricBasketPayoff
from sde.random_process import RandomProcess
from pricer.pricing_result import PricingResult
from pricer.american.longstaff_schwartz import LongstaffSchwartz


class LongstaffSchwartzVR(LongstaffSchwartz):
    """
    Longstaff-Schwartz with variance reduction for Bermudan basket options.

    Inherits the core algorithm from LongstaffSchwartz.
    Adds price_antithetic(), price_with_cv(), and price_full_vr().

    Parameters
    ----------
    regression_degree : int
        Polynomial degree for the continuation value regression. Default: 2.
    """

    # ------------------------------------------------------------------
    # 1. Antithetic variates
    # ------------------------------------------------------------------

    def price_antithetic(
        self,
        bermudan_payoff: BermudanBasketPayoff,
        process: RandomProcess,
        nb_paths: int,
        nb_steps: int,
        rate: float,
    ) -> PricingResult:
        """
        Longstaff-Schwartz with antithetic variates.

        Algorithm
        ---------
        1. Simulate M paths forward (Z paths) via _simulate_paths().
        2. Compute M antithetic paths analytically:
              log(S_anti_{t+1}/S_anti_t)_i = 2*(r_i - σ_i²/2)*dt - log(S_{t+1}/S_t)_i
           This exploits the identity Phi^{-1}(1-u) = -Phi^{-1}(u) reflected
           in the log-normal increments. No extra RNG calls needed.
        3. Run LS backward induction on all 2M paths together
           (richer regression sample).
        4. Average payoffs in (Z_j, -Z_j) pairs before computing PricingResult,
           so the estimator variance correctly reflects the negative correlation.

        Parameters
        ----------
        bermudan_payoff : BermudanBasketPayoff
        process : RandomProcess  (BSMilsteinND, pseudo-random or QMC)
        nb_paths : int
        nb_steps : int
        rate : float

        Returns
        -------
        PricingResult  (nb_paths reported = M, i.e. number of pairs)
        """
        T, _, n_assets, K, weights, dt, exercise_steps = \
            self._parse_payoff(bermudan_payoff, nb_steps)

        # Retrieve per-asset parameters needed for the antithetic formula
        rates_list, vols_list = self._get_rates_vols(process, n_assets)

        # Step 1: simulate M original paths (with QMC detection)
        z_values = self._simulate_paths(process, nb_paths, n_assets, T, nb_steps)

        # Step 2: compute M antithetic paths analytically
        anti_values = self._antithetic_paths(z_values, n_assets, nb_steps, rates_list, vols_list, dt)

        # Step 3: combine → shape (n_assets, 2M, nb_steps+1)
        combined    = np.concatenate([z_values, anti_values], axis=1)
        total_paths = 2 * nb_paths

        cashflow, tau_step = self._run_ls(
            combined, total_paths, exercise_steps, weights, K, dt, rate
        )

        # Step 4: discount, then average in (Z_j, -Z_j) pairs
        time_to_exercise = tau_step * dt
        discounted = np.exp(-rate * time_to_exercise) * cashflow

        paired = 0.5 * (discounted[:nb_paths] + discounted[nb_paths:])

        return PricingResult.from_payoffs(
            paired.tolist(), rate=0.0, T=T,
            nb_paths=nb_paths, method="Longstaff-Schwartz (Antithetic)"
        )

    # ------------------------------------------------------------------
    # 2. Control variate
    # ------------------------------------------------------------------

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
        Longstaff-Schwartz with a static control variate.

        The control is the European geometric basket call evaluated at
        maturity T on the same simulated path:
            g_j = e^{-rT} * max(G_T_j - K, 0)
        with known analytical expectation E[g] = cv_payoff.analytical_price().

        The CV estimator is:
            f_j - beta * (g_j - E[g])
        where f_j = e^{-r*tau_j} * intrinsic_at_tau_j  (LS payoff)
        and beta = Cov(f, g) / Var(g)  estimated from a pilot simulation.

        Parameters
        ----------
        bermudan_payoff : BermudanBasketPayoff
        process : RandomProcess  (pseudo-random or QMC)
        nb_paths : int
        nb_steps : int
        rate : float
        cv_payoff : GeometricBasketPayoff
            Geometric basket payoff with analytical_price() implemented.
        E_cv : float
            Undiscounted expectation of g: cv_payoff.analytical_price() * exp(r*T).
            Kept for consistency with the Q1 interface (E_cv = analytical_price * e^{rT}).
            Internally, the DISCOUNTED E[g] = cv_payoff.analytical_price() is used.
        pilot_paths : int
            Number of pilot paths to estimate beta. Default: 2000.

        Returns
        -------
        PricingResult
        """
        T, _, n_assets, K, weights, dt, exercise_steps = \
            self._parse_payoff(bermudan_payoff, nb_steps)

        E_g_disc = cv_payoff.analytical_price()   # discounted E[g] = true price

        # Pilot: estimate beta = Cov(f, g) / Var(g)
        beta = self._estimate_beta_ls(
            process, pilot_paths, n_assets, T, nb_steps,
            exercise_steps, weights, K, dt, rate,
            cv_payoff, E_g_disc
        )

        # Main simulation
        all_values          = self._simulate_paths(process, nb_paths, n_assets, T, nb_steps)
        cashflow, tau_step  = self._run_ls(all_values, nb_paths, exercise_steps, weights, K, dt, rate)

        f = np.exp(-rate * tau_step * dt) * cashflow
        g = self._geo_basket_disc(all_values, cv_payoff, n_assets, rate, T, K)

        cv_payoffs = f - beta * (g - E_g_disc)

        return PricingResult.from_payoffs(
            cv_payoffs.tolist(), rate=0.0, T=T,
            nb_paths=nb_paths, method="Longstaff-Schwartz (CV)"
        )

    # ------------------------------------------------------------------
    # 3. Full VR: QMC + Antithetic + Control Variate
    # ------------------------------------------------------------------

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
        Longstaff-Schwartz combining QMC + Antithetic variates + Control variate.

        1. Use a QMC process (SobolPathGenerator) for low-discrepancy paths.
        2. Compute antithetic (-Z) paths analytically for each Z path.
        3. Run LS on all 2M paths together.
        4. Apply CV correction to each of the 2M discounted payoffs.
        5. Average in (Z_j, -Z_j) pairs:
               combined_j = 0.5 * (cv_z_j + cv_anti_j)

        Parameters
        ----------
        bermudan_payoff : BermudanBasketPayoff
        process : RandomProcess
            Must use a SobolPathGenerator in its generator chain.
        nb_paths : int
        nb_steps : int
        rate : float
        cv_payoff : GeometricBasketPayoff
        E_cv : float
            E_cv = cv_payoff.analytical_price() * exp(r*T)  (undiscounted).
        pilot_paths : int
            Paths used to estimate beta. Default: 2000.

        Returns
        -------
        PricingResult  (nb_paths = M, number of Z/-Z pairs)
        """
        T, _, n_assets, K, weights, dt, exercise_steps = \
            self._parse_payoff(bermudan_payoff, nb_steps)

        rates_list, vols_list = self._get_rates_vols(process, n_assets)
        E_g_disc = cv_payoff.analytical_price()

        # Pilot: estimate beta using 2*pilot_paths paths (Z + anti)
        beta = self._estimate_beta_ls_anti(
            process, pilot_paths, n_assets, T, nb_steps,
            exercise_steps, weights, K, dt, rate,
            cv_payoff, E_g_disc, rates_list, vols_list
        )

        # Simulate M original paths (QMC: next_path() called automatically)
        z_values    = self._simulate_paths(process, nb_paths, n_assets, T, nb_steps)
        anti_values = self._antithetic_paths(z_values, n_assets, nb_steps, rates_list, vols_list, dt)

        # Run LS on 2M paths
        combined    = np.concatenate([z_values, anti_values], axis=1)
        total_paths = 2 * nb_paths
        cashflow, tau_step = self._run_ls(
            combined, total_paths, exercise_steps, weights, K, dt, rate
        )

        # Discount LS payoffs
        f = np.exp(-rate * tau_step * dt) * cashflow

        # Compute CV correction for all 2M paths
        g = self._geo_basket_disc(combined, cv_payoff, n_assets, rate, T, K)
        cv_all = f - beta * (g - E_g_disc)

        # Average in (Z_j, -Z_j) pairs
        cv_z    = cv_all[:nb_paths]
        cv_anti = cv_all[nb_paths:]
        paired  = 0.5 * (cv_z + cv_anti)

        return PricingResult.from_payoffs(
            paired.tolist(), rate=0.0, T=T,
            nb_paths=nb_paths, method="Longstaff-Schwartz (QMC+Anti+CV)"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

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
        Compute the antithetic path matrix from original paths.

        For the log-Euler (Milstein) scheme:
            log(S_{t+1}/S_t)_i = (r_i - σ_i²/2)*dt + σ_i*√dt*X_i
        The antithetic corresponds to -X_i, giving:
            log(S_anti_{t+1}/S_anti_t)_i = 2*(r_i - σ_i²/2)*dt - log(S_{t+1}/S_t)_i

        This correctly mirrors correlated Gaussians through Cholesky, because
        B*(-Z) = -(B*Z), so every correlated increment is simply negated.

        Parameters
        ----------
        all_values : np.ndarray, shape (n_assets, nb_paths, nb_steps+1)
        n_assets, nb_steps : int
        rates, vols : list of float  (per-asset)
        dt : float

        Returns
        -------
        anti_values : np.ndarray, same shape as all_values
        """
        anti = np.zeros_like(all_values)
        for i in range(n_assets):
            anti[i, :, 0] = all_values[i, :, 0]          # same initial spot
            drift_i = (rates[i] - 0.5 * vols[i] ** 2) * dt
            for t in range(nb_steps):
                S_t   = np.maximum(all_values[i, :, t],   1e-12)
                S_tp1 = np.maximum(all_values[i, :, t+1], 1e-12)
                log_ret      = np.log(S_tp1 / S_t)
                log_ret_anti = 2.0 * drift_i - log_ret
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
        """
        Compute discounted European geometric basket payoff at maturity
        for every path in all_values.

        g_j = e^{-rT} * max(G_T_j - K, 0)
        G_T_j = exp(Σ_i w_i * log(S_i_T_j))

        Parameters
        ----------
        all_values : np.ndarray, shape (n_assets, nb_paths, nb_steps+1)
        cv_payoff  : GeometricBasketPayoff  (provides normalized weights)
        n_assets, rate, T, K : scalars

        Returns
        -------
        g_disc : np.ndarray, shape (nb_paths,)
        """
        norm_w   = cv_payoff._norm_w         # normalized weights
        S_T      = [np.maximum(all_values[i, :, -1], 1e-12) for i in range(n_assets)]
        log_geo  = sum(norm_w[i] * np.log(S_T[i]) for i in range(n_assets))
        G_T      = np.exp(log_geo)
        disc     = math.exp(-rate * T)
        return disc * np.maximum(G_T - K, 0.0)

    def _estimate_beta_ls(
        self,
        process: RandomProcess,
        pilot_paths: int,
        n_assets: int,
        T: float,
        nb_steps: int,
        exercise_steps: List[int],
        weights: List[float],
        K: float,
        dt: float,
        rate: float,
        cv_payoff: GeometricBasketPayoff,
        E_g_disc: float,
    ) -> float:
        """
        Estimate beta = Cov(f, g) / Var(g) from a pilot simulation.

        f = discounted LS payoff at optimal exercise.
        g = discounted European geometric basket payoff at maturity.
        """
        av = self._simulate_paths(process, pilot_paths, n_assets, T, nb_steps)
        cf, tau = self._run_ls(av, pilot_paths, exercise_steps, weights, K, dt, rate)

        f_vals = np.exp(-rate * tau * dt) * cf
        g_vals = self._geo_basket_disc(av, cv_payoff, n_assets, rate, T, K)

        return self._beta_from_samples(f_vals, g_vals)

    def _estimate_beta_ls_anti(
        self,
        process: RandomProcess,
        pilot_paths: int,
        n_assets: int,
        T: float,
        nb_steps: int,
        exercise_steps: List[int],
        weights: List[float],
        K: float,
        dt: float,
        rate: float,
        cv_payoff: GeometricBasketPayoff,
        E_g_disc: float,
        rates_list: List[float],
        vols_list: List[float],
    ) -> float:
        """
        Estimate beta using antithetic pilot paths (Z + anti).
        The estimator averages each (Z, -Z) pair, consistent with price_full_vr.
        """
        z_av   = self._simulate_paths(process, pilot_paths, n_assets, T, nb_steps)
        av_anti = self._antithetic_paths(z_av, n_assets, nb_steps, rates_list, vols_list, dt)
        combined = np.concatenate([z_av, av_anti], axis=1)

        cf, tau = self._run_ls(combined, 2 * pilot_paths, exercise_steps, weights, K, dt, rate)

        f_all = np.exp(-rate * tau * dt) * cf
        g_all = self._geo_basket_disc(combined, cv_payoff, n_assets, rate, T, K)

        # CV correction on each path, then pair-average (consistent with price_full_vr)
        f_paired = 0.5 * (f_all[:pilot_paths] + f_all[pilot_paths:])
        g_paired = 0.5 * (g_all[:pilot_paths] + g_all[pilot_paths:])

        return self._beta_from_samples(f_paired, g_paired)

    @staticmethod
    def _beta_from_samples(f: np.ndarray, g: np.ndarray) -> float:
        """beta = Cov(f, g) / Var(g), from numpy arrays."""
        n    = len(f)
        mf, mg = f.mean(), g.mean()
        cov   = float(np.sum((f - mf) * (g - mg))) / (n - 1)
        var_g = float(np.sum((g - mg) ** 2)) / (n - 1)
        return cov / var_g if var_g > 0 else 1.0

    @staticmethod
    def _get_rates_vols(process: RandomProcess, n_assets: int):
        """
        Extract per-asset rates and vols from a BSMilsteinND process.
        Falls back to replicated scalar values for BSMilstein1D.
        """
        if hasattr(process, "_rates"):
            return list(process._rates), list(process._vols)
        # 1D fallback (single asset)
        return [process._rate] * n_assets, [process._vol] * n_assets
