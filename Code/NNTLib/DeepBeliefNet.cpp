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
	SoftmaxGroup = 0;
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

	Layers = new DBNLayer[LayersCount]();
	if (softmax != 0) {
		SoftmaxGroup = softmax;
		Softmax.Init(0, SoftmaxGroup);
	}
	Layers[0].Init(0, neuronsCountPerLayer[0] + 1); //Neuronen Inputlayer
	for (int i = 1; i < LayersCount; ++i) { //Rückwärtsgewichte
		if (softmax != 0 && i == LayersCount - 1) {
			Layers[i].Init(neuronsCountPerLayer[i - 1] + SoftmaxGroup, neuronsCountPerLayer[i] + 1);
		} else
			Layers[i].Init(neuronsCountPerLayer[i - 1], neuronsCountPerLayer[i] + 1);
		TotalNeuronCount += neuronsCountPerLayer[i];
	}
	for (int i = 0; i < LayersCount - 1; ++i) { //Vorwärts
		Layers[i].Forwardweightsinit(neuronsCountPerLayer[i], &Layers[i + 1]);
	}

	InitWeights(initType);
	if (softmax != 0) {
		for (int i = 0; i < SoftmaxGroup; i++) {
			Softmax.Neurons[i].ForwardWeightCount = Layers[layercount - 1].NeuronCount - 1;
			Softmax.Neurons[i].ForwardWeights = new double*[Softmax.Neurons[i].ForwardWeightCount];
			for (int j = 0; j < Layers[layercount - 1].NeuronCount - 1; j++) { //-1 da keine Gewichte zum Bias
				Softmax.Neurons[i].ForwardWeights[j] = &Layers[layercount - 1].Neurons[j].Weights[neuronsCountPerLayer[LayersCount - 2] + i + 1];

			}
		}
	}
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
	this->SoftmaxGroup = that.SoftmaxGroup;
	this->Softmax = that.Softmax;
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

/**
 * @brief Makes Binary decision
 * @details Rolls a Dice between 0 and 1 and returns 1 if x is greater than the dice
 * 
 * @param x propability to turn on
 * @return 1 if dice < x, 0 if dice > x
 */
int DeepBeliefNet::Binary(double x) {
	srand48(time(NULL));
	//std::uniform_real_distribution<double> dist(0.0, 1.0);
	double y = drand48();
	//std::cout<<x<<" "<<y<<" "<<std::endl;
	return x >= y && x != 0;
}

/**
 * @brief Propagates Data through the net
 * @details Propagates Data through the net for sigmoid Outputs, Softmax outputs and a Softmax Group
 * 
 * @param input Input Data for first layer
 */
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

		if (LastLayerFunction == FunctionEnum::SOFTMAX) {
			double nets[top->NeuronCount];
			for (int k = 0; k < top->NeuronCount; ++k) {
				nets[k] = 0.0;
				for (int j = 0; j < top->InputValuesCount; ++j) {
					nets[k] += top->Neurons[k].Weights[j] * bottom->Neurons[j].Output;
				}
			}
			for (int k = 0; k < top->NeuronCount; ++k) {
				top->Neurons[k].Output = ActivationFunction(LastLayerFunction, nets[k], nets, top->NeuronCount);
				//std::cout<<k<<" "<<nets[k]<<std::endl;
				//std::cout<<Layers[i].Neurons[k].Output<<std::endl;
			}
			//std::cout<<std::endl;
		} else {
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
		}
	} else {

		//TODO Softmax
		top = &Layers[LayersCount - 1];
		bottom = &Layers[LayersCount - 2];
		double fe[SoftmaxGroup];
		for (int s = 0; s < SoftmaxGroup; s++) {
			for (int j = 0; j < SoftmaxGroup; j++) {
				if (s == j)
					Softmax.Neurons[j].Output = 1;
				else
					Softmax.Neurons[j].Output = 0;
				
			}
			fe[s] = 0.0;

			for (int g = 0; g < 30; g++) {
				for ( int j = 0; j < top->NeuronCount - 1; j++) {
					double sum = 0.0;
					for (int i = 0; i < bottom->NeuronCount - 1; i++) {
						sum += bottom->Neurons[i].Output * top->Neurons[j].Weights[i];
					}
					for (int i = 0; i < SoftmaxGroup; i++) {
						sum += Softmax.Neurons[i].Output**Softmax.Neurons[i].ForwardWeights[j];
					}

					sum += *bottom->Neurons[bottom->NeuronCount - 1].ForwardWeights[j]; //bias
					sum = 1 + exp(-sum);
					sum = 1 / sum; //Wahrscheinlichkeit ausrechnen
					top->Neurons[j].p = sum;
					top->Neurons[j].Output = Binary(sum); //Binäre Zustände bestimmen
				}
				double sumssoft[SoftmaxGroup];
				for (int i = 0; i < SoftmaxGroup; i++) {
					sumssoft[i] = 0;
					for (int j = 0; j < top->NeuronCount - 1; j++) {
						sumssoft[i] += top->Neurons[j].Output * *Softmax.Neurons[i].ForwardWeights[j];
					}
				}
				for (int i = 0; i < SoftmaxGroup; i++) {
					double y = 0.0;
					double x = exp(sumssoft[i]);
					for (int j = 0; j < SoftmaxGroup; j++) {
						y += exp(sumssoft[j]);
					}
					Softmax.Neurons[i].p = x / y;
					int maxindex = 0;
					for (int i = 0; i < SoftmaxGroup; i++) {
						if (Softmax.Neurons[maxindex].p < Softmax.Neurons[i].p)
							maxindex = i;
						Softmax.Neurons[i].Output = 0;
					}
					Softmax.Neurons[maxindex].Output = 1;
				}

			}
			for ( int j = 0; j < bottom->NeuronCount - 1; j++) {
				fe[s] -=bottom->Neurons[s].Output*top->Neurons[top->NeuronCount-1].Weights[j];
			}
			for (int i = 0; i < SoftmaxGroup; i++) {
				fe[s]-=Softmax.Neurons[i].Output;
			}

			for(int i=0;i<top->NeuronCount-1;i++){
				double sum=0.0;
				for(int j=0; j<bottom->NeuronCount-1;j++){
					sum+=bottom->Neurons[j].Output**bottom->Neurons[j].ForwardWeights[i];
				}
				for(int j=0; j< SoftmaxGroup;j++){
					sum+=*Softmax.Neurons[j].ForwardWeights[i]*Softmax.Neurons[j].Output;
				}
				sum+=top->Neurons[top->NeuronCount-1].Weights[i];
				sum=1+exp(sum);
				
			
				fe[s]-=std::log(sum);

			}

		}
		for (int s = 0; s < SoftmaxGroup; s++) {
			Softmax.Neurons[s].Output=fe[s];
		}
	}
}
}