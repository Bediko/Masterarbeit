#include "DeepBeliefNet.h"
#include "DBNLayer.h"
#include <iostream>

namespace NNTLib
{

DeepBeliefNet::DeepBeliefNet(int *neuronsCountPerLayer, int layercount, WeightInitEnum initType, FunctionEnum functionType) {
	init();

	if (layercount < 2)
		throw std::runtime_error("We need at least 2 Layer (input and output)");
	for (int i = 0; i < LayersCount; ++i) {
		if (neuronsCountPerLayer[i] < 1)
			throw std::runtime_error("Layer need at least 1 neuron");
	}

	this->FunctionType = functionType;

	LayersCount = layercount;

	Layers = new DBNLayer[LayersCount]();
	Layers[0].Init(0, neuronsCountPerLayer[0] + 1); //Neuronen Inputlayer
	for (int i = 1; i < LayersCount; ++i) { //Rückwärtsgewichte
		Layers[i].Init(neuronsCountPerLayer[i - 1], neuronsCountPerLayer[i] + 1);
		TotalNeuronCount += neuronsCountPerLayer[i];
	}
	for (int i = 0; i < LayersCount - 1; ++i) { //Vorwärts
		Layers[i].Forwardweightsinit(neuronsCountPerLayer[i], &Layers[i + 1]);
	}
	InitWeights(initType);
}

void DeepBeliefNet::copy(const DeepBeliefNet &that) {
	this->LayersCount = that.LayersCount;
	this->MeanSquareError = that.MeanSquareError;
	this->TotalNeuronCount = that.TotalNeuronCount;
	this->WeightInitType = that.WeightInitType;
	this->FunctionType = that.FunctionType;

	this->Layers = new DBNLayer[LayersCount];

	for (int i = 0; i < LayersCount; i++)
		this->Layers[i] = that.Layers[i];
}	

DeepBeliefNet::DeepBeliefNet(const DeepBeliefNet &that)
// :NeuralNetwork() funktioniert unter g++ jedoch bekommt der windows compiler das leider nicht hin deshalb workaround mit init()
{
	init();
	copy(that);
}

void DeepBeliefNet::InitWeights(WeightInitEnum initType) {
	this->WeightInitType = initType;

	int i, j;
	for (i = (LayersCount - 1); i >= 0; i--) {
		DBNLayer* layer = &Layers[i];

		for (j = 0; j < layer->NeuronCount; ++j) {
			DBNNeuron* neuron = &layer->Neurons[j];

			for (int k = 0; k < layer->InputValuesCount + 1; ++k) { //+1 for Bias
				if (neuron->WeightCount == 0)
					continue;
				neuron->Weights[k] = GenerateRandomWeight(layer->InputValuesCountWithBias);
			}
		}
	}
}


}