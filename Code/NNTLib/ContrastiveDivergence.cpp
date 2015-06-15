#include "ContrastiveDivergence.h"
#include <iostream>
#include "DataContainer.h"
#include <omp.h>

namespace NNTLib {


void ContrastiveDivergence::GibbsSampling(int gibbssteps, int d_i) {

	for (int g = 0; g < gibbssteps; g++) {
		//std::cout<<"Gibbs Step:"<<g<<"\r";
		//reconstruct visible units

		//use probabilities in last update to reduce sampling noise
		UpdateVisibleUnits();
		UpdateHiddenUnits();
	}
}



void ContrastiveDivergence::UpdateHiddenUnits() {

#pragma omp parallel for
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

void ContrastiveDivergence::UpdateVisibleUnits() {

#pragma omp parallel for
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

}


int ContrastiveDivergence::Binary(double x) {
	std::uniform_real_distribution<double> dist(0.0, 1.0);
	double y = dist(generator);
	//std::cout<<x<<" "<<y<<" "<<std::endl;
	return x >= y;
}

void ContrastiveDivergence::trainIncremental(const DataContainer &container, const double learnRate, int Epochs, int gibbssteps) {
	std::cout << "train CD incremental" << std::endl;
	//int gibbssteps = 20;
	//Epochs = 10;
	std::cout << "LERNRATE: " << learnRate << std::endl;
	std::cout << "EPOCHEN: " << Epochs << std::endl;
	std::cout << "GIBBS: " << gibbssteps << std::endl;





	DataContainer *input = new DataContainer[network->LayersCount](); //Container for input data on the layers
	DataContainer *biasdata;
	//Container für Initialisierung von Bias Knoten
	input[0] = container;
	//create datacontainers for every layer

	for ( int l = 1; l < network->LayersCount - 1; l++) {
		input[l].Init(input[l - 1].DataCount, network->Layers[l].NeuronCount - 1, network->Layers[l + 1].NeuronCount - 1);
	}
	input[network->LayersCount - 1].Init(input[network->LayersCount - 2].DataCount, network->Layers[network->LayersCount - 1].NeuronCount, 0);
	//RBM für jedes Layerpaar von unten nach oben
	for ( int l = 0; l < network->LayersCount - 1; l++) {
		std::cout << "Layer:" << l << " Epoche:" << 0 << " Datensatz: " << 0 << "\r" << std::flush;
		biasdata = new DataContainer();

		bottom = &network->Layers[l]; //bottom RBM Layer
		top = &network->Layers[l + 1]; //top RBM Layer
		statisticsdatav = new double*[input[l].DataCount];
		statisticsdatah = new double *[input[l].DataCount];
		//Matrix für Statistik
		statisticsmodelv = new double*[input[l].DataCount];
		statisticsmodelh = new double *[input[l].DataCount];

		biasdata->Init(input[l].DataCount, bottom->NeuronCount - 1, top->NeuronCount - 1);
		bottom->Neurons[bottom->NeuronCount - 1].InitBias(NULL); //initialise bias for hidden units
		// for (int d_i = 0; d_i < input[l].DataCount; d_i++) { //determine which Units are on for bias
		// 	for (int i = 0; i < bottom->NeuronCount - 1; i++) {
		// 		biasdata->DataInput[d_i][i] = Binary(input[l].DataInput[d_i][i]);
		//	}
		// }
		top->Neurons[top->NeuronCount - 1].InitBias(&input[l]); //initialise bias for visible units
		for (int i = 0; i < bottom->NeuronCount - 1; i++) {
			//std::cout<<top->Neurons[top->NeuronCount - 1].Weights[i]<<std::endl;
		}

		//Initialisieren des Netztes mit Eingabedaten vor dem Gibbs Sampling
		for (int e = 0; e < Epochs; e++) { //jede Maschine über Epochenazahl tranieren
			std::cout << "Layer:" << l << " Epoche:" << e << " Datensatz: " << 0 << "\r" << std::flush;
			//speichern für modell statistik allokieren
			for ( int d_i = 0; d_i < input[l].DataCount; d_i++) {
				statisticsdatav[d_i] = new double[bottom->NeuronCount];
				statisticsdatah[d_i] = new double[top->NeuronCount];
				statisticsmodelv[d_i] = new double[bottom->NeuronCount];
				statisticsmodelh[d_i] = new double[top->NeuronCount];
				for (int i = 0; i < bottom->NeuronCount - 1; i++) {
					statisticsdatav[d_i][i] = 0;
					statisticsmodelv[d_i][i] = 0;
				}
				for (int j = 0; j < top->NeuronCount - 1; j++) {
					statisticsdatah[d_i][j] = 0;
					statisticsmodelh[d_i][j] = 0;
				}
			}

			for ( int d_i = 0; d_i < input[l].DataCount; d_i++) { //Alle Trainingsdaten für jede Maschine
				for (int j = 0; j < bottom->NeuronCount - 1; ++j) { //Eingabedaten setzen
					bottom->Neurons[j].p = input[l].DataInput[d_i][j]; //Interpret data as probability to turn on
					bottom->Neurons[j].Output = Binary(bottom->Neurons[j].p);
				}
				std::cout << "Layer:" << l << " Epoche:" << e << " Datensatz: " << d_i << "\r" << std::flush;
				//Versteckte Neuronen berechnen
				UpdateHiddenUnits();
				//sammle Statistikdaten von Eingabedaten v*h
				//sammle Statistik für v_data
				for (int i = 0; i < bottom->NeuronCount - 1; i++) {
					statisticsdatav[d_i][i] = bottom->Neurons[i].p;
				}

				for (int j = 0; j < top->NeuronCount - 1; j++) {
					statisticsdatah[d_i][j] = top->Neurons[j].p;
				}

				//Gibbs Sampling
				GibbsSampling(gibbssteps, d_i);
				//sammle Statistikdaten von Modell v*h

				//sammle Statistik für visible modell
				//UpdateHiddenUnitsWithSampling();
				for (int i = 0; i < bottom->NeuronCount - 1; i++) {
					statisticsmodelv[d_i][i] = bottom->Neurons[i].p;
				}
				//hidden
				for (int j = 0; j < top->NeuronCount - 1; j++) {
					statisticsmodelh[d_i][j] = top->Neurons[j].p;
				}
				double edata = 0.0, emodel = 0.0;

				for (int i = 0; i < bottom->NeuronCount - 1; i++) {
					edata = 0.0;
					emodel = 0.0;
					for ( int j = 0; j < top->NeuronCount - 1; j++) {

						//for ( int d_i = 0; d_i < input[l].DataCount; d_i++) {

						emodel = statisticsmodelv[d_i][i] * statisticsmodelh[d_i][j];
						edata = statisticsdatav[d_i][i] * statisticsdatah[d_i][j];
						//}
						//emodel /= (double)input[l].DataCount;
						//edata /= (double)input[l].DataCount;
						//Gewichte anpassen

						//std::cout << edata << " " << emodel << std::endl;
						*bottom->Neurons[i].ForwardWeights[j] += learnRate * (edata - emodel);
					}

				}


				//bias anpassen für visible units
				for ( int i = 0; i < bottom->NeuronCount - 1; i++) {
					emodel = 0.0;
					edata = 0.0;
					emodel = statisticsmodelv[d_i][i];
					edata = statisticsdatav[d_i][i];

					top->Neurons[top->NeuronCount - 1].Weights[i] += learnRate * (edata - emodel);
				}
				//bias anpassen für hidden units
				for ( int j = 0; j < top->NeuronCount - 1; j++) {
					emodel = 0.0;
					edata = 0.0;
					//for ( int d_i = 0; d_i < input[l].DataCount; d_i++) {
					emodel = statisticsmodelh[d_i][j];
					edata = statisticsdatah[d_i][j];
					//}
					*bottom->Neurons[bottom->NeuronCount - 1].ForwardWeights[j] += learnRate * (edata - emodel);
				}


			} // Ende Trainingssatz




			delete biasdata;
			biasdata = nullptr;

			for ( int d_i = 0; d_i < input[l].DataCount; d_i++) {
				delete [] statisticsdatav[d_i];
				delete [] statisticsmodelv[d_i];
				delete [] statisticsdatah[d_i];
				delete [] statisticsmodelh[d_i];

			}

		}
		for ( int d_i = 0; d_i < input[l].DataCount; d_i++) {
			for (int j = 0; j < bottom->NeuronCount - 1; ++j) { //Eingabedaten setzen
				bottom->Neurons[j].p = input[l].DataInput[d_i][j]; //Interpret data as probability to turn on
			}
			//Versteckte Neuronen berechnen
			UpdateHiddenUnits();
			for ( int j = 0; j < top->NeuronCount - 1; j++) {
				input[l + 1].DataInput[d_i][j] = top->Neurons[j].p;
			}


		}

		delete [] statisticsdatav;
		delete [] statisticsmodelv;
		delete [] statisticsdatah;
		delete [] statisticsmodelh;
	}
	delete [] input;


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
