#pragma once
#include "PseudoGenerator.h"
#include "LinearCongruential.h"

class EcuyerCombined : public PseudoGenerator
{
protected:
	LinearCongruential FirstLinear;
	LinearCongruential SecondLinear;
public:
	EcuyerCombined();
	~EcuyerCombined();

	virtual double Generate();
};

