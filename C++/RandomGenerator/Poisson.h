#pragma once
#include "DiscreteGenerator.h"
class Poisson : public DiscreteGenerator
{
protected:
	double Lambda;
	myLong Factorial(myLong n);

public:
	Poisson(double lambda, UniformGenerator * unif);
	~Poisson();
};

class PoissonFirstAlgo : public Poisson
{
public:
	PoissonFirstAlgo(double lambda, UniformGenerator* unif);
	double Generate();
};

class PoissonSecondAlgo : public Poisson
{
public:
	PoissonSecondAlgo(double lambda, UniformGenerator* unif);
	double Generate();
};
