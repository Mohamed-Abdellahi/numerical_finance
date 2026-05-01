"""
Tests for the payoff module
============================
Run with:
    python3 -m pytest tests/test_payoff.py -v

Tests cover:
  - VanillaCallPayoff / VanillaPutPayoff : structure + MC price vs BS formula
  - BasketCallPayoff : weights, multi-asset, put-call parity
  - BermudanBasketPayoff : intrinsic values, exercise dates structure
"""

import math
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from random_generator import NormalBoxMuller, EcuyerCombined
from sde import BSMilstein1D, BSMilsteinND
from payoff import (
    VanillaCallPayoff, VanillaPutPayoff,
    BasketCallPayoff, BasketPutPayoff,
    BermudanBasketPayoff,
)


# ---------------------------------------------------------------------------
# Black-Scholes closed-form (benchmark)
# ---------------------------------------------------------------------------

def bs_call_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)

def bs_put_price(S, K, T, r, sigma):
    call = bs_call_price(S, K, T, r, sigma)
    # Put-Call Parity: P = C - S + K*e^{-rT}
    return call - S + K * math.exp(-r * T)

def _ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))

def mc_price(payoff_obj, process, nb_paths, T, nb_steps, rate):
    """Generic Monte Carlo pricer for European payoffs."""
    payoffs = []
    for _ in range(nb_paths):
        process.simulate(0.0, T, nb_steps)
        payoffs.append(payoff_obj(process.get_all_paths()))
    return math.exp(-rate * T) * sum(payoffs) / nb_paths


# ===========================================================================
# 1. VANILLA CALL
# ===========================================================================

class TestVanillaCallPayoff:

    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

    def test_positive_payoff_itm(self):
        """Deep ITM call: payoff must be positive."""
        call = VanillaCallPayoff(strike=50.0)
        gen  = NormalBoxMuller(0.0, 1.0, EcuyerCombined())
        proc = BSMilstein1D(gen, spot=200.0, rate=0.05, vol=0.1)
        proc.simulate(0.0, 1.0, 100)
        assert call(proc.get_all_paths()) > 0.0

    def test_zero_payoff_otm(self):
        """Deep OTM call always returns exactly 0 (very high strike)."""
        call = VanillaCallPayoff(strike=10_000.0)
        gen  = NormalBoxMuller(0.0, 1.0, EcuyerCombined())
        proc = BSMilstein1D(gen, spot=100.0, rate=0.05, vol=0.2)
        proc.simulate(0.0, 1.0, 100)
        assert call(proc.get_all_paths()) == 0.0

    def test_mc_price_vs_bs(self):
        """Monte Carlo call price must match Black-Scholes within 2%."""
        bs_ref = bs_call_price(self.S0, self.K, self.T, self.r, self.sigma)
        call   = VanillaCallPayoff(strike=self.K)
        gen    = NormalBoxMuller(0.0, 1.0, EcuyerCombined())
        proc   = BSMilstein1D(gen, self.S0, self.r, self.sigma)
        mc     = mc_price(call, proc, nb_paths=30_000, T=self.T,
                          nb_steps=100, rate=self.r)
        assert abs(mc - bs_ref) / bs_ref < 0.02, \
            f"MC={mc:.4f}, BS={bs_ref:.4f}"

    def test_wrong_number_of_paths(self):
        """Passing 2 paths to a vanilla payoff should raise ValueError."""
        call = VanillaCallPayoff(strike=100.0)
        gen  = NormalBoxMuller(0.0, 1.0, EcuyerCombined())
        proc = BSMilsteinND(gen, spots=[100.0, 100.0],
                            rates=[0.05, 0.05], vols=[0.2, 0.2])
        proc.simulate(0.0, 1.0, 50)
        with pytest.raises(ValueError):
            call(proc.get_all_paths())

    def test_invalid_strike(self):
        with pytest.raises(ValueError):
            VanillaCallPayoff(strike=-10.0)


# ===========================================================================
# 2. VANILLA PUT
# ===========================================================================

class TestVanillaPutPayoff:

    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

    def test_positive_payoff_itm(self):
        """Deep ITM put: payoff must be positive."""
        put  = VanillaPutPayoff(strike=200.0)
        gen  = NormalBoxMuller(0.0, 1.0, EcuyerCombined())
        proc = BSMilstein1D(gen, spot=50.0, rate=0.05, vol=0.1)
        proc.simulate(0.0, 1.0, 100)
        assert put(proc.get_all_paths()) > 0.0

    def test_mc_price_vs_bs(self):
        """Monte Carlo put price must match Black-Scholes within 2%."""
        bs_ref = bs_put_price(self.S0, self.K, self.T, self.r, self.sigma)
        put    = VanillaPutPayoff(strike=self.K)
        gen    = NormalBoxMuller(0.0, 1.0, EcuyerCombined())
        proc   = BSMilstein1D(gen, self.S0, self.r, self.sigma)
        mc     = mc_price(put, proc, nb_paths=30_000, T=self.T,
                          nb_steps=100, rate=self.r)
        assert abs(mc - bs_ref) / bs_ref < 0.02, \
            f"MC={mc:.4f}, BS={bs_ref:.4f}"

    def test_put_call_parity(self):
        """Put-Call Parity: C - P = S0 - K*e^{-rT}."""
        call_payoff = VanillaCallPayoff(strike=self.K)
        put_payoff  = VanillaPutPayoff(strike=self.K)

        # Shared generator so call and put see same paths
        gen  = NormalBoxMuller(0.0, 1.0, EcuyerCombined())
        proc = BSMilstein1D(gen, self.S0, self.r, self.sigma)

        nb_paths = 30_000
        call_payoffs, put_payoffs = [], []
        for _ in range(nb_paths):
            proc.simulate(0.0, self.T, 100)
            paths = proc.get_all_paths()
            call_payoffs.append(call_payoff(paths))
            put_payoffs.append(put_payoff(paths))

        discount = math.exp(-self.r * self.T)
        mc_call = discount * sum(call_payoffs) / nb_paths
        mc_put  = discount * sum(put_payoffs)  / nb_paths

        parity_lhs = mc_call - mc_put
        parity_rhs = self.S0 - self.K * math.exp(-self.r * self.T)

        assert abs(parity_lhs - parity_rhs) < 0.5, \
            f"Put-Call parity violated: C-P={parity_lhs:.4f}, S-Ke^-rT={parity_rhs:.4f}"


# ===========================================================================
# 3. BASKET CALL
# ===========================================================================

class TestBasketCallPayoff:

    def test_reduces_to_vanilla_single_asset(self):
        """Basket with 1 asset weight=1 equals vanilla call."""
        K  = 100.0
        S0, r, sigma, T = 105.0, 0.05, 0.2, 1.0

        basket = BasketCallPayoff(weights=[1.0], strike=K)
        vanilla = VanillaCallPayoff(strike=K)

        gen  = NormalBoxMuller(0.0, 1.0, EcuyerCombined())
        proc = BSMilstein1D(gen, S0, r, sigma)

        nb_paths = 20_000
        b_payoffs, v_payoffs = [], []
        for _ in range(nb_paths):
            proc.simulate(0.0, T, 100)
            paths = proc.get_all_paths()
            b_payoffs.append(basket(paths))
            v_payoffs.append(vanilla(paths))

        discount = math.exp(-r * T)
        mc_basket  = discount * sum(b_payoffs)  / nb_paths
        mc_vanilla = discount * sum(v_payoffs) / nb_paths

        assert abs(mc_basket - mc_vanilla) < 0.01, \
            f"Basket(w=[1.0]) != Vanilla: {mc_basket:.4f} vs {mc_vanilla:.4f}"

    def test_symmetric_basket(self):
        """
        Equal-weight basket of 2 identical assets with rho=1
        should price like a single-asset call.
        """
        S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
        bs_ref = bs_call_price(S0, K, T, r, sigma)

        basket = BasketCallPayoff(weights=[0.5, 0.5], strike=K)
        gen    = NormalBoxMuller(0.0, 1.0, EcuyerCombined())
        proc   = BSMilsteinND(gen, spots=[S0, S0], rates=[r, r],
                              vols=[sigma, sigma],
                              correlation_matrix=[[1.0, 1.0], [1.0, 1.0]])

        mc = mc_price(basket, proc, nb_paths=20_000, T=T, nb_steps=100, rate=r)
        assert abs(mc - bs_ref) / bs_ref < 0.03, \
            f"Symmetric basket MC={mc:.4f}, BS={bs_ref:.4f}"

    def test_payoff_non_negative(self):
        """All basket payoffs must be >= 0."""
        basket = BasketCallPayoff(weights=[0.5, 0.5], strike=100.0)
        gen    = NormalBoxMuller(0.0, 1.0, EcuyerCombined())
        proc   = BSMilsteinND(gen, spots=[100.0, 100.0],
                              rates=[0.05, 0.05], vols=[0.2, 0.3])
        for _ in range(200):
            proc.simulate(0.0, 1.0, 50)
            assert basket(proc.get_all_paths()) >= 0.0

    def test_wrong_number_of_paths(self):
        basket = BasketCallPayoff(weights=[0.5, 0.5], strike=100.0)
        gen    = NormalBoxMuller(0.0, 1.0, EcuyerCombined())
        proc   = BSMilstein1D(gen, 100.0, 0.05, 0.2)
        proc.simulate(0.0, 1.0, 50)
        with pytest.raises(ValueError):
            basket(proc.get_all_paths())   # 1 path instead of 2

    def test_basket_value_at_step(self):
        """basket_value_at_step must return a float at any valid step."""
        basket = BasketCallPayoff(weights=[0.6, 0.4], strike=100.0)
        gen    = NormalBoxMuller(0.0, 1.0, EcuyerCombined())
        proc   = BSMilsteinND(gen, spots=[100.0, 90.0],
                              rates=[0.05, 0.05], vols=[0.2, 0.25])
        proc.simulate(0.0, 1.0, 100)
        v = basket.basket_value_at_step(proc.get_all_paths(), step=50)
        assert isinstance(v, float) and v > 0.0


# ===========================================================================
# 4. BASKET PUT
# ===========================================================================

class TestBasketPutPayoff:

    def test_payoff_non_negative(self):
        put  = BasketPutPayoff(weights=[0.5, 0.5], strike=100.0)
        gen  = NormalBoxMuller(0.0, 1.0, EcuyerCombined())
        proc = BSMilsteinND(gen, spots=[100.0, 100.0],
                            rates=[0.05, 0.05], vols=[0.2, 0.3])
        for _ in range(200):
            proc.simulate(0.0, 1.0, 50)
            assert put(proc.get_all_paths()) >= 0.0


# ===========================================================================
# 5. BERMUDAN BASKET PAYOFF
# ===========================================================================

class TestBermudanBasketPayoff:

    def test_terminal_payoff_matches_basket(self):
        """Bermudan terminal payoff == BasketCallPayoff terminal payoff."""
        K, weights = 100.0, [0.5, 0.5]
        ex_dates = [0.0, 0.25, 0.5, 0.75, 1.0]

        berm   = BermudanBasketPayoff(weights=weights, strike=K,
                                      exercise_dates=ex_dates)
        basket = BasketCallPayoff(weights=weights, strike=K)

        gen  = NormalBoxMuller(0.0, 1.0, EcuyerCombined())
        proc = BSMilsteinND(gen, spots=[100.0, 100.0],
                            rates=[0.05, 0.05], vols=[0.2, 0.3])
        for _ in range(100):
            proc.simulate(0.0, 1.0, 100)
            paths = proc.get_all_paths()
            assert berm(paths) == basket(paths)

    def test_intrinsic_value_non_negative(self):
        """Intrinsic value at any step must be >= 0."""
        berm = BermudanBasketPayoff(weights=[0.5, 0.5], strike=100.0,
                                    exercise_dates=[0.0, 0.5, 1.0])
        gen  = NormalBoxMuller(0.0, 1.0, EcuyerCombined())
        proc = BSMilsteinND(gen, spots=[100.0, 100.0],
                            rates=[0.05, 0.05], vols=[0.2, 0.3])
        proc.simulate(0.0, 1.0, 100)
        for step in [0, 25, 50, 75, 100]:
            assert berm.intrinsic_value(proc.get_all_paths(), step) >= 0.0

    def test_exercise_dates_properties(self):
        ex_dates = [0.0, 0.25, 0.5, 0.75, 1.0]
        berm = BermudanBasketPayoff(weights=[1.0], strike=100.0,
                                    exercise_dates=ex_dates)
        assert berm.maturity == 1.0
        assert berm.n_exercise_dates == 5
        assert berm.exercise_dates == ex_dates

    def test_invalid_exercise_dates_not_increasing(self):
        with pytest.raises(ValueError):
            BermudanBasketPayoff(weights=[1.0], strike=100.0,
                                 exercise_dates=[0.0, 1.0, 0.5])

    def test_invalid_single_exercise_date(self):
        with pytest.raises(ValueError):
            BermudanBasketPayoff(weights=[1.0], strike=100.0,
                                 exercise_dates=[1.0])

    def test_bermudan_more_expensive_than_european(self):
        """
        A Bermudan option is always >= European (more rights = more value).
        We verify this approximately via MC using intrinsic values.
        """
        K, T = 100.0, 1.0
        r, sigma = 0.05, 0.2
        ex_dates = [0.0, 0.25, 0.5, 0.75, 1.0]
        weights  = [1.0]

        european = VanillaCallPayoff(strike=K)
        berm     = BermudanBasketPayoff(weights=weights, strike=K,
                                        exercise_dates=ex_dates)

        gen  = NormalBoxMuller(0.0, 1.0, EcuyerCombined())
        proc = BSMilstein1D(gen, spot=100.0, rate=r, vol=sigma)

        nb_paths = 10_000
        eur_payoffs, berm_payoffs = [], []

        # Step indices corresponding to exercise dates (100 steps total over T=1)
        exercise_steps = [0, 25, 50, 75, 100]

        for _ in range(nb_paths):
            proc.simulate(0.0, T, 100)
            paths = proc.get_all_paths()

            # European: terminal value only
            eur_payoffs.append(european(paths))

            # Bermudan: best intrinsic value across all exercise dates
            # (upper bound — not the true LS price, just checking dominance)
            berm_payoffs.append(max(
                berm.intrinsic_value(paths, step=s)
                for s in exercise_steps
            ))

        discount = math.exp(-r * T)
        mc_eur  = discount * sum(eur_payoffs)  / nb_paths
        mc_berm = discount * sum(berm_payoffs) / nb_paths

        # Bermudan (with early exercise) >= European
        assert mc_berm >= mc_eur - 0.5, \
            f"Bermudan {mc_berm:.4f} < European {mc_eur:.4f}"
