"""
Tests — pricer module (Schlogl architecture)
==============================================
Architecture inspired by Schlogl (2014): MCEngine + MCMapping decorators.

Run with:
    python3 -m pytest tests/test_pricer.py -v
"""

import math
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from random_generator import NormalBoxMuller, EcuyerCombined
from random_generator.sobol import SobolGenerator
from sde import BSMilstein1D, BSMilsteinND
from payoff import (
    VanillaCallPayoff, VanillaPutPayoff,
    BasketCallPayoff, GeometricBasketPayoff, BermudanBasketPayoff,
)
from pricer import (
    PricingResult, MCMapping,
    BasicMapping, AntitheticMapping, ControlVariateMapping,
    MCEngine, LongstaffSchwartz,
)


# ---------------------------------------------------------------------------
# Black-Scholes closed-form helpers
# ---------------------------------------------------------------------------

def bs_call(S, K, T, r, sigma):
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    N  = lambda x: 0.5*(1 + math.erf(x/math.sqrt(2)))
    return S*N(d1) - K*math.exp(-r*T)*N(d2)

def bs_put(S, K, T, r, sigma):
    return bs_call(S, K, T, r, sigma) - S + K*math.exp(-r*T)

def make_proc_1d(S0=100.0, r=0.05, sigma=0.2):
    return BSMilstein1D(NormalBoxMuller(0.0,1.0,EcuyerCombined()), S0, r, sigma)

def make_proc_nd(spots=[100.0], rates=[0.05], vols=[0.2], corr=None):
    return BSMilsteinND(NormalBoxMuller(0.0,1.0,EcuyerCombined()),
                        spots, rates, vols, corr)


# ===========================================================================
# 1. PricingResult
# ===========================================================================

class TestPricingResult:

    def test_from_payoffs(self):
        r = PricingResult.from_payoffs([10.0,12.0,8.0,11.0], 0.05, 1.0, 4, "Test")
        assert r.price > 0
        assert r.std_error > 0
        assert r.conf_interval[0] < r.price < r.conf_interval[1]

    def test_ci_width(self):
        r = PricingResult(10.5, 0.1, (10.3, 10.7), 1000, "T")
        assert r.ci_width == pytest.approx(0.4, abs=1e-9)


# ===========================================================================
# 2. BasicMapping + MCEngine  (standard Monte Carlo)
# ===========================================================================

class TestBasicMapping:

    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

    def test_call_price_close_to_bs(self):
        bs_ref  = bs_call(self.S0, self.K, self.T, self.r, self.sigma)
        mapping = BasicMapping(VanillaCallPayoff(self.K), make_proc_1d())
        result  = MCEngine().price(mapping, nb_paths=30_000, T=self.T,
                                   nb_steps=100, rate=self.r)
        assert abs(result.price - bs_ref) / bs_ref < 0.02

    def test_put_price_close_to_bs(self):
        bs_ref  = bs_put(self.S0, self.K, self.T, self.r, self.sigma)
        mapping = BasicMapping(VanillaPutPayoff(self.K), make_proc_1d())
        result  = MCEngine().price(mapping, nb_paths=30_000, T=self.T,
                                   nb_steps=100, rate=self.r)
        assert abs(result.price - bs_ref) / bs_ref < 0.02

    def test_ci_covers_bs(self):
        bs_ref  = bs_call(self.S0, self.K, self.T, self.r, self.sigma)
        mapping = BasicMapping(VanillaCallPayoff(self.K), make_proc_1d())
        result  = MCEngine().price(mapping, nb_paths=30_000, T=self.T,
                                   nb_steps=100, rate=self.r)
        lo, hi = result.conf_interval
        assert lo <= bs_ref <= hi

    def test_invalid_nb_paths(self):
        mapping = BasicMapping(VanillaCallPayoff(self.K), make_proc_1d())
        with pytest.raises(ValueError):
            MCEngine().price(mapping, nb_paths=1, T=self.T, nb_steps=10, rate=self.r)

    def test_invalid_maturity(self):
        mapping = BasicMapping(VanillaCallPayoff(self.K), make_proc_1d())
        with pytest.raises(ValueError):
            MCEngine().price(mapping, nb_paths=100, T=0.0, nb_steps=10, rate=self.r)

    def test_is_mc_mapping(self):
        assert isinstance(BasicMapping(VanillaCallPayoff(100.0), make_proc_1d()), MCMapping)


# ===========================================================================
# 3. LongstaffSchwartz (Bermudan)
# ===========================================================================

class TestLongstaffSchwartz:

    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

    def test_bermudan_price_positive(self):
        berm   = BermudanBasketPayoff([1.0], self.K, [0.0,0.25,0.5,0.75,1.0])
        result = LongstaffSchwartz().price(berm, make_proc_nd(),
                                           nb_paths=5_000, nb_steps=100, rate=self.r)
        assert result.price > 0.0

    def test_bermudan_geq_european(self):
        bs_ref = bs_call(self.S0, self.K, self.T, self.r, self.sigma)
        berm   = BermudanBasketPayoff([1.0], self.K, [0.0,0.25,0.5,0.75,1.0])
        result = LongstaffSchwartz().price(berm, make_proc_nd(),
                                           nb_paths=10_000, nb_steps=100, rate=self.r)
        assert result.price >= bs_ref - 0.5

    def test_result_has_ci(self):
        berm   = BermudanBasketPayoff([1.0], self.K, [0.0,0.5,1.0])
        result = LongstaffSchwartz().price(berm, make_proc_nd(),
                                           nb_paths=2_000, nb_steps=50, rate=self.r)
        assert result.conf_interval[0] < result.price < result.conf_interval[1]

    def test_multi_asset(self):
        proc   = make_proc_nd([100.0,80.0],[0.05,0.05],[0.2,0.3],[[1.0,0.4],[0.4,1.0]])
        berm   = BermudanBasketPayoff([0.5,0.5], 90.0, [0.0,0.5,1.0])
        result = LongstaffSchwartz().price(berm, proc,
                                           nb_paths=5_000, nb_steps=50, rate=0.05)
        assert result.price > 0.0


# ===========================================================================
# 4. GeometricBasketPayoff (analytical)
# ===========================================================================

class TestGeometricBasketPayoff:

    def test_analytical_price_close_to_bs(self):
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
        geo = GeometricBasketPayoff([1.0], K, [S], T, r, [sigma], [[1.0]])
        assert abs(geo.analytical_price() - bs_call(S, K, T, r, sigma)) < 0.1

    def test_positive_price(self):
        geo = GeometricBasketPayoff([1/3]*3, 100.0, [100.0,90.0,110.0], 1.0, 0.05,
                                    [0.2,0.3,0.15],
                                    [[1.0,0.3,0.2],[0.3,1.0,0.4],[0.2,0.4,1.0]])
        assert geo.analytical_price() > 0.0

    def test_payoff_on_paths(self):
        geo  = GeometricBasketPayoff([1.0,1.0], 90.0, [100.0,100.0], 1.0, 0.05, [0.2,0.2])
        proc = make_proc_nd([120.0,120.0],[0.05,0.05],[0.001,0.001])
        proc.simulate(0.0, 0.001, 1)
        assert geo(proc.get_all_paths()) > 0.0


# ===========================================================================
# 5. AntitheticMapping
# ===========================================================================

class TestAntitheticMapping:

    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

    def test_price_close_to_bs(self):
        bs_ref  = bs_call(self.S0, self.K, self.T, self.r, self.sigma)
        mapping = AntitheticMapping(make_proc_nd(), BasketCallPayoff([1.0], self.K))
        result  = MCEngine().price(mapping, nb_paths=20_000, T=self.T,
                                   nb_steps=100, rate=self.r)
        assert abs(result.price - bs_ref) / bs_ref < 0.02

    def test_lower_std_error_than_basic(self):
        payoff   = BasketCallPayoff([1.0], self.K)
        basic    = BasicMapping(payoff, make_proc_nd())
        anti     = AntitheticMapping(make_proc_nd(), payoff)
        r_basic  = MCEngine().price(basic, nb_paths=10_000, T=self.T,
                                    nb_steps=100, rate=self.r)
        r_anti   = MCEngine().price(anti,  nb_paths=5_000,  T=self.T,
                                    nb_steps=100, rate=self.r)
        assert r_anti.std_error <= r_basic.std_error * 1.2

    def test_is_mc_mapping(self):
        assert isinstance(AntitheticMapping(make_proc_nd(), VanillaCallPayoff(100.0)),
                          MCMapping)


# ===========================================================================
# 6. ControlVariateMapping
# ===========================================================================

class TestControlVariateMapping:

    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

    def test_price_close_to_bs(self):
        bs_ref     = bs_call(self.S0, self.K, self.T, self.r, self.sigma)
        proc       = make_proc_nd()
        payoff     = BasketCallPayoff([1.0], self.K)
        geo        = GeometricBasketPayoff([1.0], self.K, [self.S0], self.T,
                                           self.r, [self.sigma], [[1.0]])
        E_cv       = geo.analytical_price() * math.exp(self.r * self.T)
        mapping    = ControlVariateMapping(proc, payoff, geo, E_cv,
                                           T_pilot=self.T, nb_steps_pilot=100)
        result     = MCEngine().price(mapping, nb_paths=10_000, T=self.T,
                                      nb_steps=100, rate=self.r)
        assert abs(result.price - bs_ref) / bs_ref < 0.02

    def test_beta_estimated(self):
        proc    = make_proc_nd()
        payoff  = BasketCallPayoff([1.0], self.K)
        geo     = GeometricBasketPayoff([1.0], self.K, [self.S0], self.T,
                                        self.r, [self.sigma], [[1.0]])
        E_cv    = geo.analytical_price() * math.exp(self.r * self.T)
        mapping = ControlVariateMapping(proc, payoff, geo, E_cv,
                                        T_pilot=self.T, nb_steps_pilot=50,
                                        pilot_paths=500)
        assert mapping.beta > 0

    def test_is_mc_mapping(self):
        proc    = make_proc_nd()
        payoff  = BasketCallPayoff([1.0], self.K)
        geo     = GeometricBasketPayoff([1.0], self.K, [self.S0], self.T,
                                        self.r, [self.sigma], [[1.0]])
        E_cv    = geo.analytical_price() * math.exp(self.r * self.T)
        mapping = ControlVariateMapping(proc, payoff, geo, E_cv,
                                        T_pilot=self.T, nb_steps_pilot=50)
        assert isinstance(mapping, MCMapping)
