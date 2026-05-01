#include "PseudoGenerator.h"

PseudoGenerator::PseudoGenerator(myLong inputSeed) :
	Seed(inputSeed),
	Current(inputSeed)
{
}

PseudoGenerator::PseudoGenerator() 
{ 
	Seed = 0; 
	Current = 0; 
}