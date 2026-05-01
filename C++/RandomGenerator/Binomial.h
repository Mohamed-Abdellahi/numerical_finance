#pragma once
#include "DiscreteGenerator.h"
#include "Bernoulli.h"

class Binomial : public DiscreteGenerator
{
protected:
	myLong n;
	Bernoulli Bern;

public:
	Binomial(myLong n, double p, UniformGenerator* unif);
	~Binomial();
	double Generate();
};

