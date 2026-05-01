#pragma once
#include "RandomGenerator.h"
#include "UniformGenerator.h"

class ContinuousGenerator : public RandomGenerator
{
protected:
	UniformGenerator* Uniform;

public:
	ContinuousGenerator(double mean, double variance, UniformGenerator* unif);
	~ContinuousGenerator();
};

