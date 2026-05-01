#include "Bernoulli.h"

Bernoulli::Bernoulli(double inputProb, UniformGenerator* unif) : p(inputProb)
{
	Uniform = unif;
}


Bernoulli::~Bernoulli()
{

}

double Bernoulli::Generate()
{
	return (Uniform->Generate() <= p);
}

