#include "HeadTail.h"

HeadTail::HeadTail(UniformGenerator* unif)
{
	Uniform = unif;
}


HeadTail::~HeadTail()
{
}

double HeadTail::Generate()
{
	return (Uniform->Generate() <= 0.5);
}