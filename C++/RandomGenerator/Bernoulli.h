#pragma once
#include "DiscreteGenerator.h"

class Bernoulli :  public DiscreteGenerator
{
protected:
	double p;

public:
	Bernoulli(double inputP, UniformGenerator* unif);
	~Bernoulli();
	double Generate();
};

