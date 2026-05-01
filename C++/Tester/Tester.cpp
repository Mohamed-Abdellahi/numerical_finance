#include <iostream>
#include <LinearCongruential.h>
#include <EcuyerCombined.h>
#include <FiniteSet.h>
#include <Exponential.h>
#include <Normal.h>
#include <Poisson.h>
#include <PDEGrid2D.h>

void TestRandom();
void TestPDE();

int main()
{
    //TestRandom(); 
    TestPDE();
}

void TestRandom()
{
    myLong Multiplier, Increment, Modulus, Seed;

    Multiplier = 40014;
    Increment = 0;
    Modulus = 2147483563;
    Seed = 1;

    //X = new LinearCongruential(Multiplier, Increment, Modulus, Seed);
    //X = new EcuyerCombined();
    UniformGenerator* Unif = new EcuyerCombined();
    std::vector<double> probs(3);
    probs[0] = 0.4;
    probs[1] = 0.5;
    probs[2] = 0.1;
    RandomGenerator* Finit = new FiniteSet(probs, Unif);
    /*
        for (size_t i = 0; i < 10; ++i)
        {
            std::cout << Finit->Generate() << std::endl;
        }

        ExponentialInverseDistribution* Exp = new ExponentialInverseDistribution(0.5, Unif);
        */
    bool isAverageOk, isVarianceOk;
    //isAverageOk = Exp->TestMean(100000, 0.01);
    //isVarianceOk = Exp->TestVariance(100000, 0.1);

    NormalBoxMuller* NormBox = new NormalBoxMuller(0., 1., Unif);
    isAverageOk = NormBox->TestMean(100000, 0.01);
    isVarianceOk = NormBox->TestVariance(100000, 0.1);
}

void TestPDE()
{
    /*
        Evaluate a Call option in the Black-Scholes framework:
        Expected Price : 6.80495771
    */
    double Spot = 100;
    double Strike = 100;
    double Maturity = 1;
    double Rate = 0.05;
    double Volatility = 0.1;

    double SMin = 0;
    double SMax = 1000;

    R2R1Function* VarianceFunction = new BSVariance(Volatility);
    R2R1Function* TrendFunction = new BSTrend(Rate);
    R2R1Function* ActualizationFunction = new BSActualization(Rate);
    R2R1Function* SourceTermFunction = new NullFunction();

    R1R1Function* TopBoundaryFunction = new CallTopBoundary(SMax, Strike);
    R1R1Function* BottomBoundaryFunction = new CallBottomBoundary(SMin, Strike);
    R1R1Function* RightBoundaryFunction = new CallTerminalCondition(Strike);

    PDEGrid2DExplicit BlackScholesGrid(Maturity, SMin, SMax, 10000, 1.,
        VarianceFunction, TrendFunction, ActualizationFunction, SourceTermFunction,
        TopBoundaryFunction, BottomBoundaryFunction, RightBoundaryFunction);

    BlackScholesGrid.FillNodes();
    double priceAtZero = BlackScholesGrid.GetTimeZeroNodeValue(Spot);

    std::cout << "Price = " << priceAtZero << std::endl;
}
