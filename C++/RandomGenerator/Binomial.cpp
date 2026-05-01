#include "Binomial.h"

Binomial::Binomial(myLong n, double p, UniformGenerator* unif) : n(n), Bern(p, unif)
{
}


Binomial::~Binomial()
{
}

double Binomial::Generate()
{
	double result = 0;

	for (myLong i = 0; i < n; ++i)
	{
		result += Bern.Generate();
	}
	return result;
}