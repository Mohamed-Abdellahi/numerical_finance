#pragma once
#include "DiscreteGenerator.h"
class HeadTail :  public DiscreteGenerator
{
public:
	HeadTail(UniformGenerator* unif);
	~HeadTail();
	double Generate();
};

