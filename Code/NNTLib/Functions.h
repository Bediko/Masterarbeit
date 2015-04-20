#ifndef FUNCTIONS_H
#define FUNCTIONS_H
//#include <math.h>
#include <cmath>
#include <random>
//#define LOGISTIC(x) (1./(1.+exp(-x)))
//#define LOGISTIC_DERIVATION(y) (y * (1. - y))

//#define TANH(x) tanh(x)
//#define TANH_DERIVATION(y) (1. - (y*y))
//
//#define LECUN_TANH(x) (1.7159*tanh(0.66666667*x))
//#define LECUN_TANH_DERIVATION(y)  (0.66666667/1.7159*(1.7159+(y))*(1.7159-(y)))

#include "Enums.h"
namespace NNTLib
{
	double ActivationFunction(FunctionEnum funcEnum, double y);

	double ActivationFunctionDerivate(FunctionEnum funcEnum, double y);
}

#endif
