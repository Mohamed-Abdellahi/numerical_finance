#include "Poisson.h"

Poisson::Poisson(double lambda, UniformGenerator* unif) : Lambda(lambda), DiscreteGenerator(lambda, lambda, unif)
{
	if (lambda <= 0)
		throw std::exception("Lambda must be strictly positive for Poisson distribution");
	Lambda = lambda;
}

Poisson::~Poisson()
{
}

myLong Poisson::Factorial(myLong n)
{
	return (n == 1 || n == 0) ? 1 : Factorial(n - 1) * n;
}

PoissonFirstAlgo::PoissonFirstAlgo(double lambda, UniformGenerator* unif) : Poisson(lambda, unif)
{}

double PoissonFirstAlgo::Generate()
{
	throw std::exception("The first algorithm is not implemented yet for Poisson distribution");
}

PoissonSecondAlgo::PoissonSecondAlgo(double lambda, UniformGenerator* unif) : Poisson(lambda, unif)
{}

double PoissonSecondAlgo::Generate()
{
	throw std::exception("The second algorithm is not implemented yet for Poisson distribution");
}