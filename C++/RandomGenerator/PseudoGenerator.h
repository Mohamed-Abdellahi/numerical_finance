#pragma once
#include "UniformGenerator.h"
class PseudoGenerator : public UniformGenerator
{
protected:
	myLong Seed;
	myLong Current;

public:
	PseudoGenerator(myLong inputSeed);
	PseudoGenerator();
	~PseudoGenerator() {}
	myLong GetCurrent() { return Current; }
};

