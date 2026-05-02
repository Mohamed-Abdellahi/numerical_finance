"""
random_generator — Package
===========================
Hierarchy of random number generators.

RandomGenerator  (ABC)
├── UniformGenerator  (ABC)                    uniform_generator.py
│   ├── PseudoGenerator  (ABC)                 pseudo_generator.py
│   │   ├── LinearCongruential                 linear_congruential.py
│   │   └── EcuyerCombined                     ecuyer_combined.py
│   ├── HaltonGenerator                        halton.py
│   └── SobolGenerator                         sobol.py
├── ContinuousGenerator  (ABC)                 continuous_generator.py
│   ├── Normal  (ABC)  ──  BoxMuller           normal.py
│   │              ──  CLT
│   │              ──  RejectionSampling
│   ├── Exponential  (ABC) ── InverseCDF       exponential.py
│   │                 ── RejectionSampling
│   └── Poisson                                poisson.py
└── DiscreteGenerator  (ABC)                   discrete_generator.py
    ├── HeadTail                               head_tail.py
    ├── Bernoulli                              bernoulli.py
    ├── Binomial                               binomial.py
    └── FiniteSet                              finite_set.py
"""

from .random_generator import RandomGenerator

# Uniform
from .uniform_generator import UniformGenerator
from .pseudo_generator import PseudoGenerator
from .linear_congruential import LinearCongruential
from .ecuyer_combined import EcuyerCombined
from .halton import HaltonGenerator
from .sobol import SobolPathGenerator

# Continuous
from .continuous_generator import ContinuousGenerator
from .normal import Normal, NormalBoxMuller, NormalCLT, NormalRejectionSampling, NormalInverseCDF
from .exponential import (
    Exponential,
    ExponentialInverseDistribution,
    ExponentialRejectionSampling,
)
from .poisson import Poisson

# Discrete
from .discrete_generator import DiscreteGenerator
from .head_tail import HeadTail
from .bernoulli import Bernoulli
from .binomial import Binomial
from .finite_set import FiniteSet

__all__ = [
    # Base
    "RandomGenerator",
    # Uniform
    "UniformGenerator", "PseudoGenerator",
    "LinearCongruential", "EcuyerCombined",
    "HaltonGenerator", "SobolGenerator",
    # Continuous
    "ContinuousGenerator",
    "Normal", "NormalBoxMuller", "NormalCLT", "NormalRejectionSampling",
    "Exponential", "ExponentialInverseDistribution", "ExponentialRejectionSampling",
    "Poisson",
    # Discrete
    "DiscreteGenerator",
    "HeadTail", "Bernoulli", "Binomial", "FiniteSet",
]
