#include "Functions.h"
namespace NNTLib
{
	/// <summary>
	/// Activations the function.
	/// </summary>
	/// <param name="funcEnum">The function enum.</param>
	/// <param name="x">The x.</param>
	/// <returns></returns>
	double ActivationFunction(FunctionEnum funcEnum, double x)
	{
		switch(funcEnum)
		{
		case FunctionEnum::LOGISTIC:
			return (1.0/(1.0 + std::exp(-x)));// LOGISTIC(x);
			// break;
		case FunctionEnum::TANH:
			return std::tanh(x);
			//break;
		case FunctionEnum::LECUN_TANH:
			return (1.7159*std::tanh(0.66666667*x));
		case FunctionEnum::LINEAR:
			return x;
			//break;
		case FunctionEnum::BINARY:
			std::default_random_engine generator;
			std::uniform_real_distribution<double> dist(0.0, 1.0);
			return x>dist(generator);
		}

		return x;//linear
	}

	/// <summary>
	/// Activations the function derivate.
	/// </summary>
	/// <param name="funcEnum">The function enum.</param>
	/// <param name="y">The y.</param>
	/// <returns></returns>
	double ActivationFunctionDerivate(FunctionEnum funcEnum, double y)
	{
		switch(funcEnum)
		{
		case FunctionEnum::LOGISTIC:
			return (y * (1.0 - y));//LOGISTIC_DERIVATION(y);
			// break;
		case FunctionEnum::TANH:
			return (1.0 - (y*y));
			//break;
		case FunctionEnum::LECUN_TANH:
			return (0.66666667/1.7159*(1.7159+(y))*(1.7159-(y)));
		case FunctionEnum::LINEAR:
			return 1;
			//break;
		}

		return 1;//linear derivation
	}
}