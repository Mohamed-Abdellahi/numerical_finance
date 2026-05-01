#include "DiscreteGenerator.h"

DiscreteGenerator::DiscreteGenerator()
{
}

DiscreteGenerator::DiscreteGenerator(double mean, double variance, UniformGenerator* unif) :
	RandomGenerator(mean, variance)
{
	Uniform = unif;
}


DiscreteGenerator::~DiscreteGenerator()
{
}
