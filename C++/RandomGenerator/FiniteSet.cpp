#include "FiniteSet.h"

FiniteSet::FiniteSet(const std::vector<double>& inputProb, UniformGenerator* uniform)
{
	Uniform = uniform;
	double sumProba = 0;
	for (size_t i = 0; i < inputProb.size(); ++i)
	{
		double currentProb = inputProb[i];
		if (currentProb < 0 || currentProb > 1)
		{
			throw std::exception("A probability must be between [0 , 1]");
		}
		else
		{
			sumProba += currentProb;
		}
	}
	if (sumProba != 1)
		throw std::exception("The sum of probabilities must be equal to 1");

	Probabilities = inputProb;
}

FiniteSet::~FiniteSet()
{
}

double FiniteSet::Generate()
{
	double result = 0.;
	double Pkminus1 = 0;
	double Pk = 0;

	double unif = Uniform->Generate();
	myLong K = 1;

	for (myLong i = 0; i < Probabilities.size(); ++i)
	{
		Pk += Probabilities[i];
		if (Pkminus1 <= unif && unif < Pk)
		{
			result = K;
			break;
		}
		else
		{
			Pkminus1 = Pk;
		}
		++K;
	}
	result = K;
	return result;
}