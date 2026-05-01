#include "ContinuousGenerator.h"

ContinuousGenerator::ContinuousGenerator(double mean, double variance, UniformGenerator* unif) : RandomGenerator(mean, variance)
{
	Uniform = unif;
}


ContinuousGenerator::~ContinuousGenerator()
{
}
