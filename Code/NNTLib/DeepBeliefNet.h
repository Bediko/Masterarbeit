#include "NeuralNetwork.h"
#include "DBNLayer.h"
#include <iostream>

#ifndef DeepBeliefNet_H
#define DeepBeliefNet_H

namespace NNTLib
{
	class DeepBeliefNet: public NeuralNetwork
	{	protected:
		void copy(const DeepBeliefNet &that);
		public:
	    DBNLayer *Layers;
		DeepBeliefNet(int *layers,int layercount,WeightInitEnum initType,FunctionEnum functionType);
		DeepBeliefNet(const DeepBeliefNet &that);
		void InitWeights(WeightInitEnum initType);

	};
}


#endif