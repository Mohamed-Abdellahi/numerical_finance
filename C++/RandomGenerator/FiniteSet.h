#pragma once
#include "DiscreteGenerator.h"
#include <vector>

class FiniteSet : public DiscreteGenerator
{
protected:
	std::vector<double> Probabilities;

public:
	FiniteSet(const std::vector<double>& inputProb, UniformGenerator* uniform);
	~FiniteSet();
	virtual double Generate();
};

