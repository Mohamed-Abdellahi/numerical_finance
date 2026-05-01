#pragma once
#include <cmath>
#include <exception>
typedef unsigned long long  myLong;

class RandomGenerator
{
private:
	const double targetMean;
	const double targetVariance;

public:
	RandomGenerator(double mean, double variance);
	RandomGenerator();
	~RandomGenerator();

	virtual double Generate() = 0;

	double Mean(myLong  nbSim);
	double Variance(myLong  nbSim);
	bool TestMean(myLong nbSim, double tol);
	bool TestVariance(myLong nbSim, double tol);
};