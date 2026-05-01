#pragma once
#include "ContinuousGenerator.h"

class Normal : public ContinuousGenerator
{
protected:
	double Mu;
	double Sigma;

public:
	Normal(double inputMu, double inputSigma, UniformGenerator* unif);
	~Normal();
};

class NormalBoxMuller : public Normal
{
private:
	bool requireNewSimulation;
	double X;
	double Y;

public:
	NormalBoxMuller(double inputMu, double inputSigma, UniformGenerator* unif);
	double Generate();
};

class NormalCLT : public Normal
{
public:
	NormalCLT(double inputMu, double inputSigma, UniformGenerator* unif);
	double Generate();
};

class NormalRejectionSampling : public Normal
{
public:
	NormalRejectionSampling(double inputMu, double inputSigma, UniformGenerator* unif);
	double Generate();
};