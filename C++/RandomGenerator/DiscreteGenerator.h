#pragma once
#include "RandomGenerator.h"
#include "UniformGenerator.h"

class DiscreteGenerator : public RandomGenerator
{
protected:

	UniformGenerator* Uniform;

public:
	DiscreteGenerator();
	DiscreteGenerator(double mean, double variance, UniformGenerator* unif);
	~DiscreteGenerator();
};

