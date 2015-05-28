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

void ContrastiveDivergence::trainIncremental(const DataContainer &container, const double learnRate, int Epochs, int gibbssteps) {
	std::cout << "train CD incremental" << std::endl;
	//int gibbssteps = 20;
	//Epochs = 10;
	std::cout << "LERNRATE: " << learnRate << std::endl;
	std::cout << "EPOCHEN: " << Epochs << std::endl;
	std::cout << "GIBBS: " << gibbssteps << std::endl;



	DataContainer *input = new DataContainer[network->LayersCount](); //Container for input data on the layers
	double ***statisticsdata; //Matrix für Statistik
	double **statisticsdatav;
	double **statisticsdatah;
	//Matrix für Statistik
	double ***statisticsmodel;
	double **statisticsmodelv;
	double **statisticsmodelh;
	//Container für Initialisierung von Bias Knoten
	input[0] = container;
	//create datacontainer with binary input to initialise Bias
	for ( int l = 1; l < network->LayersCount - 1; l++) {
		input[l].Init(input[l - 1].DataCount, network->Layers[l].NeuronCount, network->Layers[l + 1].NeuronCount);
	}
	input[network->LayersCount - 1].Init(input[network->LayersCount - 2].DataCount, network->Layers[network->LayersCount - 1].NeuronCount, 0);
	//RBM für jedes Layerpaar von unten nach oben
	for ( int l = 0; l < network->LayersCount - 1; l++) {
		DBNLayer *bottom, *top;
		DataContainer *biasdata = new DataContainer();

		bottom = &network->Layers[l]; //bottom RBM Layer


		top = &network->Layers[l + 1]; //top RBM Layer
		statisticsmodel = new double**[input[l].DataCount];
		statisticsdata = new double**[input[l].DataCount]; //Matrix für Statistik
		statisticsdatav = new double*[input[l].DataCount];
		statisticsdatah = new double * [input[l].DataCount];
		//Matrix für Statistik
		statisticsmodelv = new double*[input[l].DataCount];
		statisticsmodelh = new double *[input[l].DataCount];

		biasdata->Init(input[l].DataCount, bottom->NeuronCount - 1, top->NeuronCount - 1);
		bottom->Neurons[bottom->NeuronCount - 1].InitBias(NULL); //initialise bias for hidden units
		for (int d_i = 0; d_i < input[l].DataCount; d_i++) { //determine which Units are on for bias
			for (int i = 0; i < bottom->NeuronCount - 1; i++) {
				biasdata->DataInput[d_i][i] = Binary(input[l].DataInput[d_i][i]);
			}
		}
		top->Neurons[top->NeuronCount - 1].InitBias(biasdata); //initialise bias for visible units

		for ( int d_i = 0; d_i < input[l].DataCount; d_i++) {
			statisticsmodel[d_i] = new double*[bottom->NeuronCount];
			statisticsdata[d_i] = new double*[bottom->NeuronCount];
			statisticsdatav[d_i] = new double[bottom->NeuronCount];
			statisticsdatah[d_i] = new double[top->NeuronCount];
			statisticsmodelv[d_i] = new double[bottom->NeuronCount];
			statisticsmodelh[d_i] = new double[top->NeuronCount];
			for (int i = 0; i < bottom->NeuronCount - 1; i++) {
				statisticsmodel[d_i][i] = new double[top->NeuronCount];
				statisticsdata[d_i][i] = new double[top->NeuronCount];
			}
		}

		//Initialisieren des Netztes mit Eingabedaten vor dem Gibbs Sampling
		for (int e = 0; e < Epochs; e++) { //jede Maschine über Epochenazahl tranieren
			//speichern für modell statistik allokieren

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
				}
				//sammle Statistikdaten von Eingabedaten v*h
				for (int i = 0; i < bottom->NeuronCount - 1; i++) {
					for ( int j = 0; j < top->NeuronCount - 1; j++) {
						statisticsdata[d_i][i][j] = bottom->Neurons[i].p * top->Neurons[j].p;
					}
				}
				//sammle Statistik für v_data

				for (int i = 0; i < bottom->NeuronCount - 1; i++) {
					statisticsdatav[d_i][i] = bottom->Neurons[i].p;
				}
				for (int j = 0; j < top->NeuronCount - 1; j++) {
					statisticsdatah[d_i][j] = top->Neurons[j].p;
				}
				//Gibbs Sampling
				for (int g = 0; g < gibbssteps; g++) {

					//reconstruct visible units
					for ( int i = 0; i < bottom->NeuronCount - 1; i++) {
						double sum = 0.0;
						for (int j = 0; j < top->NeuronCount - 1; j++) {
							sum += top->Neurons[j].Output * *bottom->Neurons[i].ForwardWeights[j]; //use p for less sampling
						}

						sum += top->Neurons[top->NeuronCount - 1].Weights[i]; //bias
						sum = 1 / (1 + exp(-sum)); //Wahrscheinlichkeit ausrechnen
						bottom->Neurons[i].p = sum;
						bottom->Neurons[i].Output = Binary(sum); //Binäre Zustände bestimmen

					}

					//use probabilities in last update to reduce sampling noise
					if (g == gibbssteps - 1) {
						for ( int j = 0; j < top->NeuronCount - 1; j++) {
							double sum = 0.0;
							for (int i = 0; i < top->Neurons[j].WeightCount; i++) {
								sum += bottom->Neurons[j].p * top->Neurons[j].Weights[i];
							}

							sum += *bottom->Neurons[bottom->NeuronCount - 1].ForwardWeights[j]; //bias
							sum = 1 / (1 + exp(-sum)); //Wahrscheinlichkeit ausrechnen
							top->Neurons[j].p = sum;
							top->Neurons[j].Output = sum; //Binäre Zustände bestimmen

						}
					}
					//use binary units
					else {
						for ( int j = 0; j < top->NeuronCount - 1; j++) {
							double sum = 0.0;
							for (int i = 0; i < top->Neurons[j].WeightCount; i++) {
								sum += bottom->Neurons[j].p * top->Neurons[j].Weights[i];
							}

							sum += *bottom->Neurons[bottom->NeuronCount - 1].ForwardWeights[j]; //bias
							sum = 1 / (1 + exp(-sum)); //Wahrscheinlichkeit ausrechnen
							top->Neurons[j].p = sum;
							top->Neurons[j].Output = Binary(sum); //Binäre Zustände bestimmen
						}
					}


				}// ende Gibbs
				//sammle Statistikdaten von Modell v*h
				for (int i = 0; i < bottom->NeuronCount - 1; i++) {
					for ( int j = 0; j < top->NeuronCount - 1; j++) {
						statisticsmodel[d_i][i][j] = bottom->Neurons[i].p * top->Neurons[j].p;
					}
				}
				//sammle Statistik für visible modell

				for (int i = 0; i < bottom->NeuronCount - 1; i++) {

					statisticsmodelv[d_i][i] = bottom->Neurons[i].p;
				}
				//hidden
				for (int j = 0; j < top->NeuronCount - 1; j++) {
					statisticsmodelh[d_i][j] = top->Neurons[j].p;
				}
			} // Ende Trainingssatz
			//Erwartungswerte Berechnen für Gewichte
			double edata = 0.0, emodel = 0.0, ebiash = 0.0, ebiasv = 0.0;

			for (int i = 0; i < bottom->NeuronCount - 1; i++) {
				for ( int j = 0; j < top->NeuronCount - 1; j++) {
					edata = 0.0;
					emodel = 0.0;
					for ( int d_i = 0; d_i < input[l].DataCount; d_i++) {

						emodel += statisticsmodel[d_i][i][j];
						edata += statisticsdata[d_i][i][j];
					}
					emodel /= (double)input[l].DataCount;
					edata /= (double)input[l].DataCount;
					//Gewichte anpassen
					*bottom->Neurons[i].ForwardWeights[j] += learnRate * (edata - emodel);
					//std::cout << edata << " " << emodel << std::endl;
				}
			}
			//bias anpassen für visible units
			for ( int i = 0; i < bottom->NeuronCount - 1; i++) {
				emodel = 0.0;
				edata = 0.0;
				for ( int d_i = 0; d_i < input[l].DataCount; d_i++) {
					emodel += statisticsmodelv[d_i][i];
					edata += statisticsdatav[d_i][i];
				}
				top->Neurons[top->NeuronCount - 1].Weights[i] += learnRate * (edata - emodel);
			}
			//bias anpassen für hidden units
			for ( int j = 0; j < top->NeuronCount - 1; j++) {
				emodel = 0.0;
				edata = 0.0;
				for ( int d_i = 0; d_i < input[l].DataCount; d_i++) {
					emodel += statisticsmodelh[d_i][j];
					edata += statisticsdatah[d_i][j];
				}
				*bottom->Neurons[bottom->NeuronCount - 1].ForwardWeights[j] += learnRate * (edata - emodel);
			}

		}
		for ( int d_i = 0; d_i < input[l].DataCount; d_i++) {
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
				input[l + 1].DataInput[d_i][j] = sum;

			}

		}
	}





	// for ( int i = 0; i < network->LayersCount; i++) {
	// 	std::cout << "Layer " << i << ": Neuronen:" << network->Layers[i].NeuronCount << std::endl;
	// 	std::cout << " Inputs: " << network->Layers[i].InputValuesCountWithBias << std::endl;
	// 	std::cout << " Gewichte:" << std::endl;
	// 	for ( int j = 0; j < network->Layers[i].NeuronCount; j++) {
	// 		DBNNeuron n = network->Layers[i].Neurons[j];
	// 		std::cout << "Neuron " << j << std::endl;
	// 		for (int k = 0; k < n.WeightCount; k++) {
	// 			std::cout << n.Weights[k] << std::endl;
	// 		}

	// 	}
	// }
	// std::cout << "Forwardweights" << std::endl;

	// for ( int i = 0; i < network->LayersCount; i++) {
	// 	std::cout << "Layer " << i << ": Neuronen:" << network->Layers[i].NeuronCount << std::endl;
	// 	std::cout << " Inputs: " << network->Layers[i].InputValuesCountWithBias << std::endl;
	// 	std::cout << " Gewichte:" << std::endl;
	// 	for ( int j = 0; j < network->Layers[i].NeuronCount; j++) {
	// 		DBNNeuron n = network->Layers[i].Neurons[j];
	// 		std::cout << "Neuron " << j << std::endl;
	// 		for (int k = 0; k < n.ForwardWeightCount; k++) {
	// 			std::cout << "Gewicht " << k << " : " << *n.ForwardWeights[k] << std::endl;
	// 		}

	// 	}
	// }
//double deltaWeight;

}
ContrastiveDivergence::ContrastiveDivergence(DeepBeliefNet & net) {
	this->network = &net;
	generator.seed(time(NULL));

}
ContrastiveDivergence::~ContrastiveDivergence() {};
void ContrastiveDivergence::Train(const DataContainer & container, const double learnRate, const int Epochs, int BatchSize, int gibbs) {
	std::cout << "Train CD" << std::endl;
	if (BatchSize == 1) {
		trainIncremental(container, learnRate, Epochs, gibbs);
		return;
	}
}
}
