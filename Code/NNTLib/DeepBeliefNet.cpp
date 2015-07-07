#include "DeepBeliefNet.h"
#include "DBNLayer.h"
#include <iostream>
#include <random>
#include <cmath>
#include <cstdlib>

namespace NNTLib {

void DeepBeliefNet::freeMem() {
	delete [] Layers;
}

/// <summary>
/// Initializes this instance.
/// </summary>
void DeepBeliefNet::init() {
	//Use random_device to generate a seed for Mersenne twister engine.
	std::random_device rd;
	// Use Mersenne twister engine to generate pseudo-random numbers.
	generator.seed(rd());

	LayersCount = 0;
	MeanSquareError = 0;

	WeightInitType = WeightInitEnum::NONE;
	FunctionType = FunctionEnum::LINEAR;
	TotalNeuronCount = 0;
	Layers = nullptr;
}
/**
 * @brief destructor
 * @details destructor
 */
DeepBeliefNet::~DeepBeliefNet() {
	freeMem();
}
/**
 * @brief Constructor
 * @details Initiliases the Deep Belief Net and builds it
 *
 * @param neuronsCountPerLayer Array with the number of Neurons for every layer
 * @param layercount Number of Layers
 * @param initType Which way to initialise Weights should be used
 * @param functionType Which activation function is used in the neurons
 */
DeepBeliefNet::DeepBeliefNet(int *neuronsCountPerLayer, int layercount, WeightInitEnum initType, FunctionEnum functionType, int softmax): NNTLib::NeuralNetwork(neuronsCountPerLayer, layercount, initType, functionType, LastLayerFunction) {
	init();

	if (layercount < 2)
		throw std::runtime_error("We need at least 2 Layer (input and output)");
	for (int i = 0; i < LayersCount; ++i) {
		if (neuronsCountPerLayer[i] < 1)
			throw std::runtime_error("Layer need at least 1 neuron");
	}

	this->FunctionType = functionType;

	LayersCount = layercount;
	if (softmax) {
		LayersCount -= 1; //Ausgabelayer kommt auf vorletzten Layer
		SoftmaxGroup = neuronsCountPerLayer[layercount];
		neuronsCountPerLayer[layercount - 1] += neuronsCountPerLayer[layercount];
	}
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
/**
 * @brief copies one Net into another
 *
 * @param that Net to copy
 */
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
/**
 * @brief Copy Constructor
 * @param that Net to copy
 */
DeepBeliefNet::DeepBeliefNet(const DeepBeliefNet &that) {
	init();
	copy(that);
}
/**
 * @brief Initialises Weights
 * @details Initialises backward Weights for every Neuron in the net
 *
 * @param initType Which type of intialisation should be used
 */
void DeepBeliefNet::InitWeights(WeightInitEnum initType) {
	this->WeightInitType = initType;

	int i, j;
	for (int i = 0; i < LayersCount; i++) {
		DBNLayer* layer = &Layers[i];

		for (j = 0; j < layer->NeuronCount - 1; ++j) {
			DBNNeuron* neuron = &layer->Neurons[j];

			for (int k = 0; k < neuron->WeightCount; ++k) { //+1 for Bias
				if (neuron->WeightCount == 0)
					continue;
				neuron->Weights[k] = GenerateRandomWeight(layer->InputValuesCountWithBias);
			}
		}
	}
}
/**
 * @brief Saves Weights for neural network
 * @details Save weights in a way that a normal feedforward net can easily read them. The backward
 * bias weights get lost in the process
 *
 * @param file Path to file that gets the weights
 */
void DeepBeliefNet::SaveWeightsforNN(const std::string file) {

	std::ofstream myfile;
	myfile.open(file.c_str());

	if (!myfile) {
		std::string buf("Could not open file");
		buf.append(file);
		throw std::runtime_error(buf);
	}

	for (int l = 1; l < LayersCount - 1; l++) {
		DBNLayer* layer = &Layers[l];
		for (int j = 0; j < layer->NeuronCount - 1; ++j) {
			DBNNeuron* neuron = &layer->Neurons[j];

			for (int i = 0; i < Layers[l - 1].NeuronCount; i++) {
				myfile << neuron->Weights[i] << "\n";
			}
		}
		for (int j = 0; j < Layers[LayersCount - 1].NeuronCount - 1; j++) {
			for (int i = 0; i < Layers[LayersCount - 2].NeuronCount; i++) {
				myfile << Layers[LayersCount - 1].Neurons[j].Weights[i] << "\n";
			}
		}

	}

	myfile.close();
}

int DeepBeliefNet::Binary(double x) {
	srand48(time(NULL));
	//std::uniform_real_distribution<double> dist(0.0, 1.0);
	double y = drand48();
	//std::cout<<x<<" "<<y<<" "<<std::endl;
	return x >= y && x != 0;
}

void DeepBeliefNet::Propagate(const double *input) {
	for (int i = 0; i < Layers[0].InputValuesCount; ++i) {
		//inputlayer hat keine neuronen daher daten in input vektor des ersten layers mit Neuronen übernehmen
		Layers[0].InputValues[i] = Binary(input[i]);
	}
	DBNLayer *top, *bottom;
	for (int i = 0; i < LayersCount - 2; ++i) {

		top = &Layers[i + 1];
		bottom = &Layers[i];
		for ( int j = 0; j < top->NeuronCount - 1; j++) {
			double sum = 0.0;
			for (int i = 0; i < bottom->NeuronCount - 1; i++) {
				sum += bottom->Neurons[i].Output * top->Neurons[j].Weights[i];
			}

			sum += *bottom->Neurons[bottom->NeuronCount - 1].ForwardWeights[j]; //bias
			sum = 1 + exp(-sum);
			sum = 1 / sum; //Wahrscheinlichkeit ausrechnen
			top->Neurons[j].p = sum;
			top->Neurons[j].Output = Binary(sum); //Binäre Zustände bestimmen
		}
	}
	if (SoftmaxGroup == 0) {
		top = &Layers[LayersCount - 1];
		bottom = &Layers[LayersCount - 2];
		for ( int j = 0; j < top->NeuronCount - 1; j++) {
			double sum = 0.0;
			for (int i = 0; i < bottom->NeuronCount - 1; i++) {
				sum += bottom->Neurons[i].Output * top->Neurons[j].Weights[i];
			}

			sum += *bottom->Neurons[bottom->NeuronCount - 1].ForwardWeights[j]; //bias
			sum = 1 + exp(-sum);
			sum = 1 / sum; //Wahrscheinlichkeit ausrechnen
			top->Neurons[j].Output = Binary(sum); //Binäre Zustände bestimmen
		}
	} else {
		//TODO Softmax
	}
}
}