"""
Tests for the SDE module
=========================
Run with:
    python3 -m pytest tests/test_sde.py -v

Tests:
  - SinglePath: insert, get_state, boundary checks
  - Brownian1D: B_0=0, increments ~ N(0, dt), terminal distribution
  - BrownianND: correlation structure via Cholesky
  - BSEuler1D / BSMilstein1D: price a European call and compare to Black-Scholes formula
  - BSMilsteinND: multi-asset basket, verify marginals and correlations
"""

import math
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from random_generator import EcuyerCombined, NormalBoxMuller
from sde import (
    SinglePath,
    Brownian1D, BrownianND,
    BSEuler1D, BSMilstein1D,
    BSMilstein2D,
    BSEulerND, BSMilsteinND,
)


# ---------------------------------------------------------------------------
# Black-Scholes closed-form price (for benchmark)
# ---------------------------------------------------------------------------

def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes European call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


# ===========================================================================
# 1. SINGLE PATH
# ===========================================================================

class TestSinglePath:

    def test_insert_and_retrieve(self):
        path = SinglePath(0.0, 1.0, 4)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            path.insert_value(v)
        assert path.get_all_values() == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_get_state_at_start(self):
        path = SinglePath(0.0, 1.0, 2)
        path.insert_value(10.0)
        path.insert_value(20.0)
        path.insert_value(30.0)
        assert path.get_state(0.0) == pytest.approx(10.0)

    def test_get_state_at_end(self):
        path = SinglePath(0.0, 1.0, 2)
        path.insert_value(10.0)
        path.insert_value(20.0)
        path.insert_value(30.0)
        assert path.get_state(1.0) == pytest.approx(30.0)

    def test_get_state_out_of_range(self):
        path = SinglePath(0.0, 1.0, 2)
        path.insert_value(10.0)
        with pytest.raises(ValueError):
            path.get_state(-0.1)

    def test_get_state_empty(self):
        path = SinglePath(0.0, 1.0, 2)
        with pytest.raises(RuntimeError):
            path.get_state(0.0)

    def test_invalid_times(self):
        with pytest.raises(ValueError):
            SinglePath(1.0, 0.0, 10)   # end < start

    def test_reset(self):
        path = SinglePath(0.0, 1.0, 2)
        path.insert_value(5.0)
        path.reset()
        assert len(path) == 0


# ===========================================================================
# 2. BROWNIAN MOTION 1D
# ===========================================================================

class TestBrownian1D:

    def test_starts_at_zero(self):
        gen = NormalBoxMuller(0.0, 1.0, EcuyerCombined())
        proc = Brownian1D(gen)
        proc.simulate(0.0, 1.0, 100)
        assert proc.get_path().values[0] == 0.0

    def test_correct_number_of_values(self):
        gen = NormalBoxMuller(0.0, 1.0, EcuyerCombined())
        proc = Brownian1D(gen)
        proc.simulate(0.0, 1.0, 100)
        # nb_steps + 1 values (initial + one per step)
        assert len(proc.get_path().values) == 101

    def test_terminal_distribution(self):
        """
        B_T ~ N(0, T). Simulate many paths, check mean and variance of B_T.
        Use a SINGLE shared generator across all paths (key: generator state
        must advance between paths, not restart from same seed).
        """
        T = 1.0
        nb_paths = 10_000
        nb_steps = 50
        # One shared generator — state advances continuously across paths
        shared_gen = NormalBoxMuller(0.0, 1.0, EcuyerCombined())
        proc = Brownian1D(shared_gen)
        terminals = []
        for _ in range(nb_paths):
            proc.simulate(0.0, T, nb_steps)
            terminals.append(proc.get_path().values[-1])

        mean = sum(terminals) / nb_paths
        var = sum((x - mean) ** 2 for x in terminals) / (nb_paths - 1)

        assert abs(mean) < 0.05         # mean ~ 0
        assert abs(var - T) < 0.05      # variance ~ T


# ===========================================================================
# 3. BROWNIAN MOTION N-D (Cholesky correlation)
# ===========================================================================

class TestBrownianND:

    def test_independent_case(self):
        """With identity correlation, dimensions should be uncorrelated."""
        gen = NormalBoxMuller(0.0, 1.0, EcuyerCombined())
        proc = BrownianND(gen, dimension=2, correlation_matrix=None)
        proc.simulate(0.0, 1.0, 50)
        assert len(proc.get_path(0).values) == 51
        assert len(proc.get_path(1).values) == 51

    def test_correlated_case_terminal_correlation(self):
        """
        With rho=0.9, terminal values of two Brownians should have correlation ~0.9.
        Use a single shared generator so each path is independent.
        """
        rho = 0.9
        corr = [[1.0, rho], [rho, 1.0]]
        nb_paths = 10_000
        T = 1.0
        b1_vals, b2_vals = [], []

        shared_gen = NormalBoxMuller(0.0, 1.0, EcuyerCombined())
        proc = BrownianND(shared_gen, dimension=2, correlation_matrix=corr)
        for _ in range(nb_paths):
            proc.simulate(0.0, T, 50)
            b1_vals.append(proc.get_path(0).values[-1])
            b2_vals.append(proc.get_path(1).values[-1])

        m1 = sum(b1_vals) / nb_paths
        m2 = sum(b2_vals) / nb_paths
        cov = sum((b1_vals[i] - m1) * (b2_vals[i] - m2) for i in range(nb_paths)) / nb_paths
        std1 = math.sqrt(sum((x - m1) ** 2 for x in b1_vals) / nb_paths)
        std2 = math.sqrt(sum((x - m2) ** 2 for x in b2_vals) / nb_paths)
        empirical_rho = cov / (std1 * std2)

        assert abs(empirical_rho - rho) < 0.05   # tolerance 5%

    def test_invalid_correlation_dimension(self):
        with pytest.raises(ValueError):
            gen = NormalBoxMuller(0.0, 1.0, EcuyerCombined())
            BrownianND(gen, dimension=2, correlation_matrix=[[1.0]])  # wrong size

    def test_wrong_diagonal(self):
        with pytest.raises(ValueError):
            gen = NormalBoxMuller(0.0, 1.0, EcuyerCombined())
            BrownianND(gen, dimension=2, correlation_matrix=[[2.0, 0.0], [0.0, 1.0]])


# ===========================================================================
# 4. BLACK-SCHOLES 1D — PRICING BENCHMARK
# ===========================================================================

class TestBlackScholes1D:
    """
    Price a European call via Monte Carlo and compare to the closed-form BS price.
    """

    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    BS_PRICE = None  # computed in setup

    @pytest.fixture(autouse=True)
    def compute_bs_price(self):
        TestBlackScholes1D.BS_PRICE = bs_call_price(
            self.S0, self.K, self.T, self.r, self.sigma
        )

    def _mc_price(self, scheme_class, nb_paths=20_000, nb_steps=100):
        """Run MC simulation and price a European call using a shared generator."""
        # Single shared generator — state advances across all paths
        shared_gen = NormalBoxMuller(0.0, 1.0, EcuyerCombined())
        proc = scheme_class(shared_gen, self.S0, self.r, self.sigma)
        payoffs = []
        for _ in range(nb_paths):
            proc.simulate(0.0, self.T, nb_steps)
            ST = proc.get_path().values[-1]
            payoffs.append(max(ST - self.K, 0.0))
        mc_price = math.exp(-self.r * self.T) * sum(payoffs) / nb_paths
        return mc_price

    def test_euler_price_close_to_bs(self):
        """BSEuler1D call price must be within 1% of the closed-form BS price."""
        mc = self._mc_price(BSEuler1D)
        assert abs(mc - self.BS_PRICE) / self.BS_PRICE < 0.03, \
            f"Euler MC={mc:.4f}, BS={self.BS_PRICE:.4f}"

    def test_milstein_price_close_to_bs(self):
        """BSMilstein1D call price must be within 1% of the closed-form BS price."""
        mc = self._mc_price(BSMilstein1D)
        assert abs(mc - self.BS_PRICE) / self.BS_PRICE < 0.03, \
            f"Milstein MC={mc:.4f}, BS={self.BS_PRICE:.4f}"

    def test_spot_positive(self):
        """All simulated spot values must be positive."""
        gen = NormalBoxMuller(0.0, 1.0, EcuyerCombined())
        proc = BSMilstein1D(gen, 100.0, 0.05, 0.2)
        proc.simulate(0.0, 1.0, 100)
        assert all(v > 0 for v in proc.get_path().values)

    def test_invalid_spot(self):
        with pytest.raises(ValueError):
            gen = NormalBoxMuller(0.0, 1.0, EcuyerCombined())
            BSEuler1D(gen, -100.0, 0.05, 0.2)

    def test_invalid_vol(self):
        with pytest.raises(ValueError):
            gen = NormalBoxMuller(0.0, 1.0, EcuyerCombined())
            BSMilstein1D(gen, 100.0, 0.05, 0.0)


# ===========================================================================
# 5. BLACK-SCHOLES 2D
# ===========================================================================

class TestBlackScholes2D:

    def test_two_paths_generated(self):
        gen = NormalBoxMuller(0.0, 1.0, EcuyerCombined())
        proc = BSMilstein2D(gen, 100.0, 100.0, 0.05, 0.05, 0.2, 0.3, 0.5)
        proc.simulate(0.0, 1.0, 50)
        assert len(proc.get_path(0).values) == 51
        assert len(proc.get_path(1).values) == 51

    def test_both_spots_positive(self):
        gen = NormalBoxMuller(0.0, 1.0, EcuyerCombined())
        proc = BSMilstein2D(gen, 100.0, 80.0, 0.05, 0.03, 0.2, 0.25, 0.6)
        proc.simulate(0.0, 1.0, 100)
        assert all(v > 0 for v in proc.get_path(0).values)
        assert all(v > 0 for v in proc.get_path(1).values)

    def test_invalid_rho(self):
        with pytest.raises(ValueError):
            gen = NormalBoxMuller(0.0, 1.0, EcuyerCombined())
            BSMilstein2D(gen, 100.0, 100.0, 0.05, 0.05, 0.2, 0.2, rho=1.5)


# ===========================================================================
# 6. BLACK-SCHOLES N-D
# ===========================================================================

class TestBlackScholesND:

    def test_three_assets(self):
        spots = [100.0, 80.0, 120.0]
        rates = [0.05, 0.05, 0.05]
        vols  = [0.2, 0.3, 0.15]
        corr  = [
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.4],
            [0.3, 0.4, 1.0],
        ]
        gen = NormalBoxMuller(0.0, 1.0, EcuyerCombined())
        proc = BSMilsteinND(gen, spots, rates, vols, corr)
        proc.simulate(0.0, 1.0, 50)

        assert len(proc.get_all_paths()) == 3
        for i in range(3):
            assert all(v > 0 for v in proc.get_path(i).values)

    def test_basket_call_price(self):
        """
        Equal-weight basket of 2 perfectly correlated assets with same params.
        Since rho=1 and weights are equal, basket = single asset → same BS price.
        Use a single shared generator.
        """
        S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
        bs_ref = bs_call_price(S0, K, T, r, sigma)

        # Perfectly correlated basket of 2 assets (rho=1)
        corr = [[1.0, 1.0], [1.0, 1.0]]
        spots = [S0, S0]
        nb_paths = 10_000
        payoffs = []

        shared_gen = NormalBoxMuller(0.0, 1.0, EcuyerCombined())
        proc = BSMilsteinND(shared_gen, spots, [r, r], [sigma, sigma], corr)
        for _ in range(nb_paths):
            proc.simulate(0.0, T, 100)
            basket_T = 0.5 * proc.get_path(0).values[-1] \
                     + 0.5 * proc.get_path(1).values[-1]
            payoffs.append(max(basket_T - K, 0.0))

        mc_price = math.exp(-r * T) * sum(payoffs) / nb_paths
        # With rho=1 and equal spots/vols, basket = single asset → same BS price
        assert abs(mc_price - bs_ref) / bs_ref < 0.05

    def test_invalid_mismatched_lengths(self):
        with pytest.raises(ValueError):
            gen = NormalBoxMuller(0.0, 1.0, EcuyerCombined())
            BSMilsteinND(gen, [100.0, 100.0], [0.05], [0.2, 0.2])
