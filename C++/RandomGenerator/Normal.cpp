#include "Normal.h"

const long double PI = 3.141592653589793238L;

Normal::Normal(double inputMu, double inputSigma, UniformGenerator* unif) : Mu(inputMu), ContinuousGenerator(inputMu, inputSigma* inputSigma, unif)
{
	if (inputSigma < 0)
		throw std::exception("The variance must be strictly positive for Normal distribution");
	Sigma = inputSigma;
}

Normal::~Normal()
{}

NormalBoxMuller::NormalBoxMuller(double inputMu, double inputSigma, UniformGenerator* unif) : Normal(inputMu, inputSigma, unif)
{
	requireNewSimulation = true;
}


NormalCLT::NormalCLT(double inputMu, double inputSigma, UniformGenerator* unif) : Normal(inputMu, inputSigma, unif)
{}

NormalRejectionSampling::NormalRejectionSampling(double inputMu, double inputSigma, UniformGenerator* unif) : Normal(inputMu, inputSigma, unif)
{}

double NormalBoxMuller::Generate()
{
	double result = 0.;

	if (requireNewSimulation)
	{
		double unif1 = Uniform->Generate();
		double unif2 = Uniform->Generate();
		double R = sqrt(-2 * log(unif1));
		double Theta = 2 * PI * unif2;
		X = Sigma * R * cos(Theta) + Mu;
		Y = Sigma * R * sin(Theta) + Mu;
		result = X;
		requireNewSimulation = false;
	}
	else
	{
		result = Y;
		requireNewSimulation = true;
	}

	return result;
}
double NormalCLT::Generate()
{
	throw std::exception("The Central Limit Theorem method is not implemented yet for Normal distribution");
}
double NormalRejectionSampling::Generate()
{
	throw std::exception("The rejection sampling method is not implemented yet for Normal distribution");
}