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
		std::mt19937 generator;
		void init();
		void freeMem();
		public:
	    DBNLayer *Layers;
	    ~DeepBeliefNet();
		DeepBeliefNet(int *layers,int layercount,WeightInitEnum initType,FunctionEnum functionType);
		DeepBeliefNet(const DeepBeliefNet &that);
		void InitWeights(WeightInitEnum initType);
		void SaveWeightsforNN(const std::string file);
		void Propagate(const double *input);
		int Binary(double x);
	};
}


#endif