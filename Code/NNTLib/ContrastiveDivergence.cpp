#include "ContrastiveDivergence.h"
#include <iostream>
#include "DataContainer.h"

namespace NNTLib
{



int ContrastiveDivergence::Binary(double x)
{
	std::uniform_real_distribution<double> dist(0.0, 1.0);
	double y = dist(generator);
	//std::cout<<x<<" "<<y<<" "<<std::endl;
	return x > y;
}

void ContrastiveDivergence::trainIncremental(const DataContainer &container, const double learnRate, const int Epochs)
{
	std::cout << "train CD incremental" << std::endl;

	//double *deltaWeights;
	network->Layers[0].Neurons[network->Layers[0].NeuronCount-1].InitBias(NULL);
	std::cout << network->Layers[1].Neurons[network->Layers[1].NeuronCount - 1].Weights[0] << std::endl;
	DataContainer *init = new DataContainer();
	init->Init(container.DataCount, container.InputCount, container.OutputCount);
	for (int d_i = 0; d_i < container.DataCount; d_i++) {
		for (int i = 0; i < container.InputCount; i++) {
			init->DataInput[d_i][i] = Binary(container.DataInput[d_i][i]);
		}
	}

	network->Layers[1].Neurons[network->Layers[1].NeuronCount - 1].InitBias(init);

	for ( int i = 0; i < network->LayersCount; i++) {
		std::cout << "Layer " << i << ": Neuronen:" << network->Layers[i].NeuronCount << std::endl;
		std::cout << " Inputs: " << network->Layers[i].InputValuesCountWithBias << std::endl;
		std::cout << " Gewichte:" << std::endl;
		for ( int j = 0; j < network->Layers[i].NeuronCount; j++) {
			DBNNeuron n = network->Layers[i].Neurons[j];
			std::cout << "Neuron " << j << std::endl;
			for (int k = 0; k < n.WeightCount; k++) {
				std::cout << n.Weights[k] << std::endl;
			}

		}
	}
	std::cout << "Forwardweights" << std::endl;

	for ( int i = 0; i < network->LayersCount; i++) {
		std::cout << "Layer " << i << ": Neuronen:" << network->Layers[i].NeuronCount << std::endl;
		std::cout << " Inputs: " << network->Layers[i].InputValuesCountWithBias << std::endl;
		std::cout << " Gewichte:" << std::endl;
		for ( int j = 0; j < network->Layers[i].NeuronCount; j++) {
			DBNNeuron n = network->Layers[i].Neurons[j];
			std::cout << "Neuron " << j << std::endl;
			for (int k = 0; k < n.ForwardWeightCount; k++) {
				std::cout << "Gewicht " << k << " : " << *n.ForwardWeights[k] << std::endl;
			}

		}
	}


	for (int e = 0; e < 1; e++) {
		for ( int d = 0; d < container.DataCount; d++) {
			for (int j = 0; j < network->Layers[0].InputValuesCount; ++j) {
				network->Layers[0].InputValues[j] = container.DataInput[d][j];
			}
#if DEBUG
			std::cout << "Inputvector" << std::endl;
			for (int j = 0; j < network->Layers[0].InputValuesCount; ++j) {
				std::cout << network->Layers[0].InputValues[j] << " ";
			}
			std::cout << std::endl;
#endif
			//bestimmen ob verstecke neuronen 1 oder 0 sind

			for ( int j = 0; j < network->Layers[0].NeuronCount; j++) {
				double sum = 0.0;
				for (int i = 0; i < network->Layers[0].Neurons[j].WeightCount; i++) {
					sum += network->Layers[0].InputValues[i] * network->Layers[0].Neurons[j].Weights[i];
				}
				sum+=network->Layers[0].Neurons[j].Bias; //bias
				sum = 1 / (1 + exp(-sum)); //Wahrscheinlichkeit ausrechnen
				network->Layers[0].Neurons[j].p=sum;
				network->Layers[0].Neurons[j].Output = Binary(sum); //Binäre Zustände bestimmen
				std::cout << network->Layers[0].Neurons[j].Output << std::endl;

			}

			//rekonstruktion


		}
	}

	//double deltaWeight;

}
ContrastiveDivergence::ContrastiveDivergence(DeepBeliefNet &net)
{
	this->network = &net;
	generator.seed(time(NULL));

}
ContrastiveDivergence::~ContrastiveDivergence() {};
void ContrastiveDivergence::Train(const DataContainer &container, const double learnRate, const int Epochs, int BatchSize)
{
	std::cout << "Train CD" << std::endl;
	if (BatchSize == 1) {
		trainIncremental(container, learnRate, Epochs);
		return;
	}
}
}
