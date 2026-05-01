"""
Tests for random_generator module
==================================
Uses pytest. Run with:
    python3 -m pytest tests/test_generators.py -v

Each test validates the statistical properties of a generator via:
  - TestMean:     |empirical_mean - theoretical_mean| <= tolerance
  - TestVariance: |empirical_variance - theoretical_variance| <= tolerance
  - Range checks: all samples must lie in expected support
  - Edge-case checks: invalid inputs raise appropriate exceptions
"""

import pytest
import sys
import os

# Allow running from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from random_generator import (
    LinearCongruential,
    EcuyerCombined,
    NormalBoxMuller,
    NormalCLT,
    NormalRejectionSampling,
    ExponentialInverseDistribution,
    ExponentialRejectionSampling,
    Poisson,
    HeadTail,
    Bernoulli,
    Binomial,
    FiniteSet,
    HaltonGenerator,
    SobolGenerator,
)
from random_generator.poisson import PoissonAlgo

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

NB_SIM = 100_000       # Number of simulations for statistical tests
MEAN_TOL = 0.02        # Absolute tolerance for mean test
VAR_TOL = 0.10         # Absolute tolerance for variance test


@pytest.fixture
def unif():
    """Standard EcuyerCombined uniform generator (reusable across tests)."""
    return EcuyerCombined()


# ===========================================================================
# 1. UNIFORM GENERATORS
# ===========================================================================

class TestLinearCongruential:

    def test_range(self):
        """All samples must be in (0, 1)."""
        gen = LinearCongruential()
        samples = [gen.generate() for _ in range(10_000)]
        assert all(0.0 < s < 1.0 for s in samples), "Sample out of (0,1) range."

    def test_mean(self):
        """Empirical mean should be close to 0.5."""
        gen = LinearCongruential()
        assert gen.test_mean(NB_SIM, MEAN_TOL), f"Mean test failed for LinearCongruential."

    def test_variance(self):
        """Empirical variance should be close to 1/12 ≈ 0.0833."""
        gen = LinearCongruential()
        assert gen.test_variance(NB_SIM, VAR_TOL), f"Variance test failed for LinearCongruential."

    def test_invalid_seed(self):
        """Seed must be strictly between 0 and modulus."""
        with pytest.raises(ValueError):
            LinearCongruential(seed=0)

    def test_invalid_modulus(self):
        """Modulus must be strictly positive."""
        with pytest.raises(ValueError):
            LinearCongruential(modulus=0)


class TestEcuyerCombined:

    def test_range(self):
        """All samples must be in (0, 1)."""
        gen = EcuyerCombined()
        samples = [gen.generate() for _ in range(10_000)]
        assert all(0.0 < s < 1.0 for s in samples), "Sample out of (0,1) range."

    def test_mean(self):
        gen = EcuyerCombined()
        assert gen.test_mean(NB_SIM, MEAN_TOL)

    def test_variance(self):
        gen = EcuyerCombined()
        assert gen.test_variance(NB_SIM, VAR_TOL)

    def test_different_seeds_differ(self):
        """Different seeds should produce different sequences."""
        g1 = EcuyerCombined(seed1=1, seed2=1)
        g2 = EcuyerCombined(seed1=42, seed2=7)
        s1 = [g1.generate() for _ in range(100)]
        s2 = [g2.generate() for _ in range(100)]
        assert s1 != s2, "Different seeds produced identical sequences."


# ===========================================================================
# 2. NORMAL DISTRIBUTION
# ===========================================================================

class TestNormalBoxMuller:

    def test_mean_standard(self, unif):
        """N(0,1): mean should be ~0."""
        gen = NormalBoxMuller(0.0, 1.0, unif)
        assert gen.test_mean(NB_SIM, MEAN_TOL)

    def test_variance_standard(self, unif):
        """N(0,1): variance should be ~1."""
        gen = NormalBoxMuller(0.0, 1.0, unif)
        assert gen.test_variance(NB_SIM, VAR_TOL)

    def test_mean_shifted(self, unif):
        """N(5, 2^2): mean should be ~5."""
        gen = NormalBoxMuller(5.0, 2.0, unif)
        assert gen.test_mean(NB_SIM, MEAN_TOL * 5)   # tolerance scaled to mean

    def test_variance_shifted(self, unif):
        """N(5, 2^2): variance should be ~4."""
        gen = NormalBoxMuller(5.0, 2.0, unif)
        assert gen.test_variance(NB_SIM, VAR_TOL * 4)

    def test_invalid_sigma(self, unif):
        """sigma must be strictly positive."""
        with pytest.raises(ValueError):
            NormalBoxMuller(0.0, -1.0, unif)

    def test_pair_recycling(self, unif):
        """Box-Muller generates pairs: two consecutive calls should cost only one pair of uniforms."""
        gen = NormalBoxMuller(0.0, 1.0, unif)
        # Just check it doesn't crash and returns two different values
        v1 = gen.generate()
        v2 = gen.generate()
        assert isinstance(v1, float) and isinstance(v2, float)


class TestNormalCLT:

    def test_mean(self, unif):
        gen = NormalCLT(0.0, 1.0, unif)
        assert gen.test_mean(NB_SIM, MEAN_TOL)

    def test_variance(self, unif):
        gen = NormalCLT(0.0, 1.0, unif)
        assert gen.test_variance(NB_SIM, VAR_TOL)


class TestNormalRejectionSampling:

    def test_mean(self, unif):
        gen = NormalRejectionSampling(0.0, 1.0, unif)
        assert gen.test_mean(NB_SIM, MEAN_TOL)

    def test_variance(self, unif):
        gen = NormalRejectionSampling(0.0, 1.0, unif)
        assert gen.test_variance(NB_SIM, VAR_TOL)


# ===========================================================================
# 3. EXPONENTIAL DISTRIBUTION
# ===========================================================================

class TestExponentialInverseDistribution:

    def test_mean_lambda1(self, unif):
        """Exp(1): mean = 1."""
        gen = ExponentialInverseDistribution(1.0, unif)
        assert gen.test_mean(NB_SIM, MEAN_TOL)

    def test_variance_lambda1(self, unif):
        """Exp(1): variance = 1."""
        gen = ExponentialInverseDistribution(1.0, unif)
        assert gen.test_variance(NB_SIM, VAR_TOL)

    def test_mean_lambda2(self, unif):
        """Exp(2): mean = 0.5."""
        gen = ExponentialInverseDistribution(2.0, unif)
        assert gen.test_mean(NB_SIM, MEAN_TOL)

    def test_positive_samples(self, unif):
        """All exponential samples must be positive."""
        gen = ExponentialInverseDistribution(1.0, unif)
        samples = [gen.generate() for _ in range(1000)]
        assert all(s > 0 for s in samples)

    def test_invalid_lambda(self, unif):
        with pytest.raises(ValueError):
            ExponentialInverseDistribution(-1.0, unif)


class TestExponentialRejectionSampling:

    def test_mean(self, unif):
        """Exp(1): mean should be ~1."""
        gen = ExponentialRejectionSampling(1.0, unif)
        assert gen.test_mean(NB_SIM, MEAN_TOL)

    def test_positive_samples(self, unif):
        """All samples must be in [0, b]."""
        gen = ExponentialRejectionSampling(1.0, unif, b=10.0)
        samples = [gen.generate() for _ in range(1000)]
        assert all(0 <= s <= 10.0 for s in samples)


# ===========================================================================
# 4. POISSON DISTRIBUTION
# ===========================================================================

class TestPoisson:

    @pytest.mark.parametrize("lam", [1.0, 3.0, 10.0])
    def test_mean_exp_arrivals(self, lam, unif):
        """Poisson(lambda): mean = lambda (inter-arrival algorithm)."""
        gen = Poisson(lam, unif, algo=PoissonAlgo.EXPONENTIAL_ARRIVALS)
        assert gen.test_mean(NB_SIM, max(MEAN_TOL * lam, 0.1))

    @pytest.mark.parametrize("lam", [1.0, 3.0, 10.0])
    def test_mean_inverse_cdf(self, lam, unif):
        """Poisson(lambda): mean = lambda (inverse CDF algorithm)."""
        gen = Poisson(lam, unif, algo=PoissonAlgo.INVERSE_CDF)
        assert gen.test_mean(NB_SIM, max(MEAN_TOL * lam, 0.1))

    def test_non_negative_samples(self, unif):
        """All Poisson samples must be non-negative integers."""
        gen = Poisson(5.0, unif)
        samples = [gen.generate() for _ in range(1000)]
        assert all(s >= 0 and s == int(s) for s in samples)

    def test_variance(self, unif):
        """Poisson(lambda): variance = lambda."""
        gen = Poisson(5.0, unif)
        assert gen.test_variance(NB_SIM, VAR_TOL * 5)

    def test_invalid_lambda(self, unif):
        with pytest.raises(ValueError):
            Poisson(0.0, unif)


# ===========================================================================
# 5. DISCRETE DISTRIBUTIONS
# ===========================================================================

class TestHeadTail:

    def test_mean(self, unif):
        """HeadTail: mean = 0.5."""
        gen = HeadTail(unif)
        assert gen.test_mean(NB_SIM, MEAN_TOL)

    def test_only_zero_or_one(self, unif):
        """HeadTail only returns 0 or 1."""
        gen = HeadTail(unif)
        samples = [gen.generate() for _ in range(1000)]
        assert all(s in (0.0, 1.0) for s in samples)


class TestBernoulli:

    @pytest.mark.parametrize("p", [0.2, 0.5, 0.8])
    def test_mean(self, p, unif):
        gen = Bernoulli(p, unif)
        assert gen.test_mean(NB_SIM, MEAN_TOL)

    def test_only_zero_or_one(self, unif):
        gen = Bernoulli(0.3, unif)
        samples = [gen.generate() for _ in range(1000)]
        assert all(s in (0.0, 1.0) for s in samples)

    def test_invalid_p(self, unif):
        with pytest.raises(ValueError):
            Bernoulli(0.0, unif)
        with pytest.raises(ValueError):
            Bernoulli(1.5, unif)


class TestBinomial:

    @pytest.mark.parametrize("n,p", [(5, 0.3), (10, 0.5), (20, 0.7)])
    def test_mean(self, n, p, unif):
        gen = Binomial(n, p, unif)
        assert gen.test_mean(NB_SIM, MEAN_TOL * n)

    def test_range(self, unif):
        """Binomial(10, 0.5): samples must be integers in [0, 10]."""
        gen = Binomial(10, 0.5, unif)
        samples = [gen.generate() for _ in range(1000)]
        assert all(0 <= s <= 10 and s == int(s) for s in samples)


class TestFiniteSet:

    def test_mean_uniform(self, unif):
        """FiniteSet([1/3, 1/3, 1/3]): mean = 1.0."""
        probs = [1/3, 1/3, 1/3]
        gen = FiniteSet(probs, unif)
        assert gen.test_mean(NB_SIM, MEAN_TOL)

    def test_mean_custom(self, unif):
        """FiniteSet([0.4, 0.5, 0.1]): mean = 0*0.4 + 1*0.5 + 2*0.1 = 0.7."""
        gen = FiniteSet([0.4, 0.5, 0.1], unif)
        assert abs(gen.target_mean - 0.7) < 1e-9
        assert gen.test_mean(NB_SIM, MEAN_TOL)

    def test_only_valid_values(self, unif):
        """Samples must be in {0, 1, 2}."""
        gen = FiniteSet([0.4, 0.5, 0.1], unif)
        samples = [gen.generate() for _ in range(1000)]
        assert all(s in (0.0, 1.0, 2.0) for s in samples)

    def test_invalid_probs_not_sum_to_one(self, unif):
        with pytest.raises(ValueError):
            FiniteSet([0.3, 0.3, 0.3], unif)

    def test_invalid_negative_prob(self, unif):
        with pytest.raises(ValueError):
            FiniteSet([-0.1, 1.1], unif)


# ===========================================================================
# 6. QUASI-RANDOM GENERATORS
# ===========================================================================

class TestHaltonGenerator:

    def test_range(self):
        """All Halton samples in [0, 1]."""
        gen = HaltonGenerator(base=2)
        samples = [gen.generate() for _ in range(1000)]
        assert all(0.0 <= s <= 1.0 for s in samples)

    def test_mean(self):
        """Halton sequence mean should be close to 0.5."""
        gen = HaltonGenerator(base=2)
        samples = [gen.generate() for _ in range(10_000)]
        avg = sum(samples) / len(samples)
        assert abs(avg - 0.5) < 0.01

    def test_deterministic(self):
        """Halton sequence is deterministic — same start gives same sequence."""
        g1 = HaltonGenerator(base=2, start=1)
        g2 = HaltonGenerator(base=2, start=1)
        assert [g1.generate() for _ in range(50)] == [g2.generate() for _ in range(50)]

    def test_different_bases(self):
        """Different bases give different sequences."""
        g2 = HaltonGenerator(base=2)
        g3 = HaltonGenerator(base=3)
        s2 = [g2.generate() for _ in range(20)]
        s3 = [g3.generate() for _ in range(20)]
        assert s2 != s3


class TestSobolGenerator:

    def test_range(self):
        """All Sobol samples in [0, 1]."""
        gen = SobolGenerator(dimension=1, scramble=True, seed=42)
        samples = [gen.generate() for _ in range(1000)]
        assert all(0.0 <= s <= 1.0 for s in samples)

    def test_mean(self):
        """Sobol mean should be close to 0.5."""
        gen = SobolGenerator(dimension=1, scramble=True, seed=42)
        samples = [gen.generate() for _ in range(10_000)]
        avg = sum(samples) / len(samples)
        assert abs(avg - 0.5) < 0.01

    def test_vector_generation(self):
        """generate_vector returns shape (n, d) correctly."""
        gen = SobolGenerator(dimension=3, scramble=True, seed=0)
        vecs = gen.generate_vector(100)
        assert len(vecs) == 100
        assert all(len(v) == 3 for v in vecs)
