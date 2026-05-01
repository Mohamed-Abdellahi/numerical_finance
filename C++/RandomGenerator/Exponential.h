#pragma once
#include "ContinuousGenerator.h"
class Exponential : public ContinuousGenerator
{
protected:
	double Lambda;

public:
	Exponential(double inputLambda, UniformGenerator* unif);
	~Exponential();
};

class ExponentialInverseDistribution : public Exponential
{
public:
	ExponentialInverseDistribution(double inputLambda, UniformGenerator* unif);
	double Generate();
};

class ExponentialRejectionSampling : public Exponential
{
public:
	ExponentialRejectionSampling(double inputLambda, UniformGenerator* unif);
	double Generate();
};