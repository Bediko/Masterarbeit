#include "Functions.h"
#include <iostream>
namespace NNTLib {
/// <summary>
/// Activations the function.
/// </summary>
/// <param name="funcEnum">The function enum.</param>
/// <param name="x">The x.</param>
/// <returns></returns>
double ActivationFunction(FunctionEnum funcEnum, double x, double* y, int size) {
	switch (funcEnum) {
	case FunctionEnum::LOGISTIC:
		return (1.0 / (1.0 + std::exp(-x))); // LOGISTIC(x);
	// break;
	case FunctionEnum::TANH:
		return std::tanh(x);
	//break;
	case FunctionEnum::LECUN_TANH:
		return (1.7159 * std::tanh(0.66666667 * x));
	case FunctionEnum::LINEAR:
		return x;
	//break;
	case FunctionEnum::BINARY:
		srand48(time(NULL));
		//std::default_random_engine generator;
		//generator.seed(time(NULL));
		x = (1.0 / (1.0 + std::exp(-x)));
		//std::uniform_real_distribution<double> dist(0.0, 1.0);
		return x >= drand48() && x != 0.0;
	case FunctionEnum::SOFTMAX:
			double sum = 0.0;
			double z;
			for(int i=0;i<size;i++){
				if(y[i]<-700)
					y[i]=-700;
				if(y[i]>700)
					y[i]=700;

					sum+=std::exp(y[i]);
			}
			if(x < -700)
				x=-700;
			if(x>700)
				x=700;
				z=std::exp(x);
				//if(isnan(z/sum))
					//std::cout<<"x "<<x <<" sum "<<sum<<std::endl;
		return z/sum;
		break;

	}

	return x;//linear
}

/// <summary>
/// Activations the function derivate.
/// </summary>
/// <param name="funcEnum">The function enum.</param>
/// <param name="y">The y.</param>
/// <returns></returns>
double ActivationFunctionDerivate(FunctionEnum funcEnum, double y) {
	switch (funcEnum) {
	case FunctionEnum::LOGISTIC:
		return (y * (1.0 - y));//LOGISTIC_DERIVATION(y);
	// break;
	case FunctionEnum::TANH:
		return (1.0 - (y * y));
	//break;
	case FunctionEnum::LECUN_TANH:
		return (0.66666667 / 1.7159 * (1.7159 + (y)) * (1.7159 - (y)));
	case FunctionEnum::LINEAR:
		return 1;
	//break;
	case FunctionEnum::SOFTMAX:
		return 1;//y * (1.0 - y);
	}

	return 1;//linear derivation
}
}
