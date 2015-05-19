#include "NeuralNetwork.h"
#include <iostream>

#ifndef DeepBeliefNet_H
#define DeepBeliefNet_H

namespace NNTLib
{
	class DeepBeliefNet: public NeuralNetwork
	{
		public:
		DeepBeliefNet(int *layers,int layercount,WeightInitEnum initType,FunctionEnum functionType);
		void InitWeights(WeightInitEnum initType);
	};
}


#endif