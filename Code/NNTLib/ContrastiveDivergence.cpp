#include "ContrastiveDivergence.h"
#include <iostream>
#include "DataContainer.h"

namespace NNTLib {



int ContrastiveDivergence::Binary(double x) {
	std::uniform_real_distribution<double> dist(0.0, 1.0);
	double y = dist(generator);
	//std::cout<<x<<" "<<y<<" "<<std::endl;
	return x > y;
}

void ContrastiveDivergence::trainIncremental(const DataContainer &container, const double learnRate, int Epochs) {
	std::cout << "train CD incremental" << std::endl;
	int gibbssteps = 1;
	Epochs = 1;

	network->Layers[0].Neurons[network->Layers[0].NeuronCount - 1].InitBias(NULL);
	DataContainer *input = new DataContainer[network->LayersCount](); //Container for input data on the layers
	double ***statisticsdata = new double**[container.DataCount]; //Matrix für Statistik
	double **statisticsdatav = new double*[container.DataCount];
	double **statisticsdatah = new double * [container.DataCount];
	//Container für Initialisierung von Bias Knoten
	input[0] = container;
	//create datacontainer with binary input to initialise Bias

	//RBM für jedes Layerpaar von unten nach oben
	for ( int l = 0; l < network->LayersCount - 1; l++) {
		DBNLayer *bottom, *top;
		DataContainer *biasdata = new DataContainer();
		bottom = &network->Layers[l]; //bottom RBM Layer
		top = &network->Layers[l + 1]; //top RBM Layer

		biasdata->Init(input[l].DataCount, bottom->NeuronCount - 1, top->NeuronCount - 1);
		bottom->Neurons[bottom->NeuronCount - 1].InitBias(NULL); //initialise bias for hidden units
		for (int d_i = 0; d_i < input[l].DataCount; d_i++) { //determine which Units are on for bias
			for (int i = 0; i < bottom->NeuronCount - 1; i++) {
				biasdata->DataInput[d_i][i] = Binary(input[l].DataInput[d_i][i]);
			}
		}
		top->Neurons[top->NeuronCount - 1].InitBias(biasdata); //initialise bias for visible units

		//Initialisieren des Netztes mit Eingabedaten vor dem Gibbs Sampling
		for (int e = 0; e < Epochs; e++) { //jede Maschine über Epochenazahl tranieren
			for ( int d_i = 0; d_i < input[l].DataCount; d_i++) { //Alle Trainingsdaten für jede Maschine
				for (int j = 0; j < bottom->InputValuesCount; ++j) { //Eingabedaten setzen
					bottom->Neurons[j].p = input[l].DataInput[d_i][j]; //Interpret data as probability to turn on
				}
				//Versteckte Neuronen berechnen
				for ( int j = 0; j < top->NeuronCount - 1; j++) {
					double sum = 0.0;
					for (int i = 0; i < top->Neurons[j].WeightCount; i++) {
						sum += bottom->Neurons[j].p * top->Neurons[j].Weights[i];
					}

					sum += *bottom->Neurons[bottom->NeuronCount - 1].ForwardWeights[j]; //bias
					sum = 1 / (1 + exp(-sum)); //Wahrscheinlichkeit ausrechnen
					top->Neurons[j].p = sum;
					top->Neurons[j].Output = Binary(sum); //Binäre Zustände bestimmen
					std::cout << top->Neurons[j].Output << std::endl;

				}
				//sammle Statistikdaten von Eingabedaten v*h
				for (int i = 0; i < bottom->NeuronCount - 1; i++) {
					statisticsdata[d_i]= new double*[bottom->NeuronCount];
					for ( int j = 0; j < top->NeuronCount - 1; j++) {
						statisticsdata[d_i][i] = new double[top->NeuronCount];
						statisticsdata[d_i][i][j]=bottom->Neurons[i].p * top->Neurons[j].p;
					}
				}
				//sammle Statistik für v_data
				statisticsdatav[d_i] = new double[bottom->NeuronCount];
				for(int i=0;i<bottom->NeuronCount-1;i++){
					statisticsdatav[d_i][i]=bottom->Neurons[i].p;
				}
				statisticsdatah[d_i] = new double[top->NeuronCount];
				for(int j=0;j<top->NeuronCount-1;j++){
					statisticsdatah[d_i][j]=top->Neurons[j].p;
				}
			}
		}


	}

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

			//bestimmen ob verstecke neuronen 1 oder 0 sind

			for ( int j = 0; j < network->Layers[0].NeuronCount; j++) {
				double sum = 0.0;
				for (int i = 0; i < network->Layers[0].Neurons[j].WeightCount; i++) {
					sum += network->Layers[0].InputValues[i] * network->Layers[0].Neurons[j].Weights[i];
				}
				sum += network->Layers[0].Neurons[j].Bias; //bias
				sum = 1 / (1 + exp(-sum)); //Wahrscheinlichkeit ausrechnen
				network->Layers[0].Neurons[j].p = sum;
				network->Layers[0].Neurons[j].Output = Binary(sum); //Binäre Zustände bestimmen
				//std::cout << network->Layers[0].Neurons[j].Output << std::endl;

			}

			//rekonstruktion


		}
	}

	//double deltaWeight;

}
ContrastiveDivergence::ContrastiveDivergence(DeepBeliefNet & net) {
	this->network = &net;
	generator.seed(time(NULL));

}
ContrastiveDivergence::~ContrastiveDivergence() {};
void ContrastiveDivergence::Train(const DataContainer & container, const double learnRate, const int Epochs, int BatchSize) {
	std::cout << "Train CD" << std::endl;
	if (BatchSize == 1) {
		trainIncremental(container, learnRate, Epochs);
		return;
	}
}
}
