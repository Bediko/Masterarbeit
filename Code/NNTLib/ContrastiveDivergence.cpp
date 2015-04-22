#include "ContrastiveDivergence.h"
#include <iostream>

namespace NNTLib {



int ContrastiveDivergence::Binary(double x) {
	std::uniform_real_distribution<double> dist(0.0,1.0);
	double y=dist(generator);
	//std::cout<<x<<" "<<y<<" "<<std::endl;
	return x > y;
}

void ContrastiveDivergence::trainIncremental(const DataContainer &container, const double learnRate, const int Epochs) {
	std::cout << "train CD incremental" << std::endl;
	int i, j, k, l;
	double error;
	//double *deltaWeights;
	Layer* layer;
	Neuron *neuron;
	double *weights;
	double *inputVector;
	double error_x_learnrate;
	for ( int i = 0; i < network->LayersCount; i++) {
		std::cout<<"Layer "<<i<<": Neuronen:";
		std::cout<<network->Layers[i].NeuronCount<<" Gewichte:";
		Neuron n = network->Layers[i].Neurons[0];
		std::cout << n.WeightCount << std::endl;
	}
	for ( int i = 0; i < container.DataCount; i++) {
		for (int j = 0; j < network->Layers[0].InputValuesCount; ++j) {
			//inputlayer hat keine neuronen daher daten in input vektor des ersten layers mit Neuronen übernehmen
			network->Layers[0].InputValues[j] = container.DataInput[i][j];
		}
#if DEBUG
		std::cout << "Inputvector" << std::endl;
		for (int j = 0; j < network->Layers[0].InputValuesCount; ++j) {
			std::cout << network->Layers[0].InputValues[j] << " ";
		}
		std::cout << std::endl;
#endif



		//for (int j=0;j< network->Layers[0]->NeuronCount;j++){
//
		//}
	}

	//double deltaWeight;

}
ContrastiveDivergence::ContrastiveDivergence(NeuralNetwork &net) {
	this->network = &net;
}
ContrastiveDivergence::~ContrastiveDivergence() {};
void ContrastiveDivergence::Train(const DataContainer &container, const double learnRate, const int Epochs, int BatchSize) {
	std::cout << "Train CD" << std::endl;
	if (BatchSize == 1) {
		trainIncremental(container, learnRate, Epochs);
		return;
	}
}
}
