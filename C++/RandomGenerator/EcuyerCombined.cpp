#include "EcuyerCombined.h"

EcuyerCombined::EcuyerCombined() :
	PseudoGenerator(),
	FirstLinear(40014, 0, 2147483563, 1),
	SecondLinear(40692, 0, 2147483399, 1)
{
}

EcuyerCombined::~EcuyerCombined()
{
}

double EcuyerCombined::Generate()
{
	double unif1 = FirstLinear.Generate();
	double unif2 = SecondLinear.Generate();
	
	myLong x1 = FirstLinear.GetCurrent();
	myLong x2 = SecondLinear.GetCurrent();

	myLong m1 = 2147483563;

	Current = (x1 - x2) % (m1 - 1);

	double result;

	if (Current > 0)
	{
		result = (double)Current / m1;
	}
	else if (Current < 0)
	{
		result = (double)Current / m1 + 1;
	}
	else
	{
		result = (double)(m1 - 1) / m1;
	}
	return result;
}