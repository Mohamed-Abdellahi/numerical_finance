# include "RandomGenerator.h"
#include <vector>

RandomGenerator::RandomGenerator() : targetMean(0.), targetVariance(0.)
{

}

RandomGenerator::RandomGenerator(double mean, double variance) : 
	targetMean(mean),
	targetVariance(variance)
{}

RandomGenerator::~RandomGenerator()
{

}

bool RandomGenerator::TestMean(myLong nbSim, double tol)
{
	double computedMean = Mean(nbSim);
	double spread = abs(computedMean - targetMean);

	if (spread > tol)
		return false;
	else
		return true;

	/*
	return (abs(Mean(nbSim) - targetMean) <= tol);
	*/
}

bool RandomGenerator::TestVariance(myLong nbSim, double tol)
{
	double computedVariance = Variance(nbSim);
	double spread = abs(computedVariance - targetVariance);

	if (spread > tol)
		return false;
	else
		return true;
}

double RandomGenerator::Mean(myLong nbSim)
{
	double result = 0.;
	for (myLong i = 0; i < nbSim; ++i)
	{
		result += Generate();
	}
	return result / nbSim;
}

double RandomGenerator::Variance(myLong nbSim)
{
	double result = 0.;
	double mean = 0.;

	std::vector<double> GeneratedNumbers;

	for (myLong i = 0; i < nbSim; ++i)
	{
		double generated = Generate();
		GeneratedNumbers.push_back(generated);

		mean += generated / nbSim;
		//mean += GeneratedNumbers.back()/nbSim;
	}
	
	for (myLong i = 0; i < nbSim; ++i)
	{
		result += pow((GeneratedNumbers[i] - mean) , 2);
	}
	return result / nbSim;
}