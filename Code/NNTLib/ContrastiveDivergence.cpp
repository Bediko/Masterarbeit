#include "ContrastiveDivergence.h"
#include "Backpropagation.h"
#include <iostream>
#include "DataContainer.h"
#include <omp.h>
#include <random>

namespace NNTLib {

/**
 * @brief Gibbs sampling
 * @details Makes gibssampling for specified number of steps
 * 
 * @param gibbssteps Number of Gibbs Samplings
 * @param softmax Number of softmax units
 */
void ContrastiveDivergence::GibbsSampling(int gibbssteps, int softmax) {

	for (int g = 0; g < gibbssteps; g++) {
		//std::cout<<"Gibbs Step:"<<g<<"\r";
		//reconstruct visible units

		//use probabilities in last update to reduce sampling noise
		UpdateVisibleUnits(softmax);
		UpdateHiddenUnits(softmax);
	}
}


/**
 * @brief Updates Hidden Units Output and propability
 * @details Updates Hidden Units Output and propability
 * 
 * @param softmaxcount size of softmax group
 */
void ContrastiveDivergence::UpdateHiddenUnits(int softmaxcount) {

	double sums[top->NeuronCount];
	#pragma omp parallel
	{
		#pragma omp for
		for ( int j = 0; j < top->NeuronCount - 1; j++) {
			sums[j] = 0.0;
			for (int i = 0; i < bottom->NeuronCount - 1; i++) {
				sums[j] += bottom->Neurons[i].Output * top->Neurons[j].Weights[i];
			}
			if (softmaxcount != 0) {
				for (int i = 0; i < softmaxcount; i++) {
					//std::cout<<*network->Softmax.Neurons[i].ForwardWeights[j]<<" "<<j<<std::endl;
					sums[j] += *network->Softmax.Neurons[i].ForwardWeights[j] * network->Softmax.Neurons[i].Output;
				}
			}

			sums[j] += *bottom->Neurons[bottom->NeuronCount - 1].ForwardWeights[j]; //bias

		}
		#pragma omp barrier
	}

	for ( int j = 0; j < top->NeuronCount - 1; j++) {
		sums[j] = 1 + exp(-sums[j]);
		sums[j] = 1 / sums[j]; //Wahrscheinlichkeit ausrechnen
		top->Neurons[j].p = sums[j];
		top->Neurons[j].Output = Binary(sums[j]); //Binäre Zustände bestimmen
	}

}
/**
 * @brief Updates visible Units Output and propabilites
 * @details Updates visible Units Output and propabilites
 * 
 * @param softmaxcount Size of Softmax Group
 */
void ContrastiveDivergence::UpdateVisibleUnits(int softmaxcount) {

	double sums[bottom->NeuronCount];
	#pragma omp parallel
	{
		#pragma omp for
		for ( int i = 0; i < bottom->NeuronCount - 1; i++) {
			sums[i] = 0.0;
			for (int j = 0; j < top->NeuronCount - 1; j++) {
				sums[i] += top->Neurons[j].Output * *bottom->Neurons[i].ForwardWeights[j]; //use p for less sampling
			}


			//Binäre Zustände bestimmen
		}
		#pragma omp barrier
	}
	for ( int i = 0; i < bottom->NeuronCount - 1; i++) {
		sums[i] += top->Neurons[top->NeuronCount - 1].Weights[i]; //bias
		sums[i] = 1 / (1 + exp(-sums[i])); //Wahrscheinlichkeit ausrechnen
		bottom->Neurons[i].p = sums[i];
		bottom->Neurons[i].Output = Binary(sums[i]);
	}
	if (softmaxcount != 0) {
		double sumssoft[softmaxcount];
		for (int i = 0; i < softmaxcount; i++) {
			sumssoft[i] = 0;
			for (int j = 0; j < top->NeuronCount - 1; j++) {
				sumssoft[i] += top->Neurons[j].Output * *network->Softmax.Neurons[i].ForwardWeights[j];
			}
		}
		for (int i = 0; i < softmaxcount; i++) {
			double y = 0.0;
			double x = exp(sumssoft[i]);
			for (int j = 0; j < softmaxcount; j++) {
				y += exp(sumssoft[j]);
			}
			network->Softmax.Neurons[i].p = x / y;
			
			int maxindex = 0;
			for (int i = 0; i < softmaxcount; i++) {
				if (network->Softmax.Neurons[maxindex].p < network->Softmax.Neurons[i].p)
					maxindex = i;
					network->Softmax.Neurons[i].Output=0;
			}
			network->Softmax.Neurons[maxindex].Output=1;

		}
	}
}

/**
 * @brief Makes Binary decision
 * @details Rolls a Dice between 0 and 1 and returns 1 if x is greater than the dice
 * 
 * @param x propability to turn on
 * @return 1 if dice < x, 0 if dice > x
 */
int ContrastiveDivergence::Binary(double x) {
	//srand48(time(NULL));
	std::uniform_real_distribution<double> dist(0.0, 1.0);
	double y = dist(generator);
	//double y = drand48();
	//std::cout<<x<<" "<<y<<" "<<std::endl;
	return x >= y && x != 0;
}

/**
 * @brief Trains a network with contrastive divergence
 * @details Trains a network with contrastive divergence
 * 
 * @param container Trainingdata
 * @param learnRate Learnrate
 * @param Epochs Number of epochs
 * @param BatchSize Size of batches
 * @param gibbssteps Number of Steps for Gibbs Sampling
 */
void ContrastiveDivergence::Train(const DataContainer & container, const double learnRate, const int Epochs, int BatchSize, int gibbssteps) {
	unsigned long long time64= GetTimeMs64();
	std::cout << "train CD incremental" << std::endl;
	//int gibbssteps = 20;
	//Epochs = 10;
	std::cout << "LERNRATE: " << learnRate << std::endl;
	std::cout << "EPOCHEN: " << Epochs << std::endl;
	std::cout << "GIBBS: " << gibbssteps << std::endl;


	int rest = container.DataCount % BatchSize;
	int batches = ((container.DataCount - rest) / BatchSize);

	if (rest != 0) {
		batches++;
	}
	std::cout << "Batches: " << batches << std::endl;




	DataContainer *input = new DataContainer[network->LayersCount](); //Container for input data on the layers
	DataContainer *biasdata;
	//Container für Initialisierung von Bias Knoten
	input[0] = container;
	//create datacontainers for every layer

	for ( int l = 1; l < network->LayersCount - 1; l++) {
		input[l].Init(input[l - 1].DataCount, network->Layers[l].NeuronCount - 1, container.OutputCount);
	}
	input[network->LayersCount - 1].Init(input[network->LayersCount - 2].DataCount, network->Layers[network->LayersCount - 1].NeuronCount, 0);
	//RBM für jedes Layerpaar von unten nach oben
	int layersend = network->LayersCount - 2;
	if (network->SoftmaxGroup != 0)
		layersend += 1;

	for ( int l = 0; l < layersend; l++) {
		//std::cout << "Layer:" << l << " Epoche:" << 0 << " Datensatz: " << 0 << "\r" << std::flush;
		biasdata = new DataContainer();

		bottom = &network->Layers[l]; //bottom RBM Layer
		top = &network->Layers[l + 1]; //top RBM Layer
		statisticsdatav = new double*[input[l].DataCount];
		statisticsdatah = new double *[input[l].DataCount];
		statisticsdatasm = new double *[input[l].DataCount];

		//Matrix für Statistik
		statisticsmodelv = new double*[input[l].DataCount];
		statisticsmodelh = new double *[input[l].DataCount];
		statisticsmodelsm = new double *[input[l].DataCount];

		biasdata->Init(input[l].DataCount, bottom->NeuronCount - 1, top->NeuronCount - 1);
		bottom->Neurons[bottom->NeuronCount - 1].InitBias(NULL); //initialise bias for hidden units
		// for (int d_i = 0; d_i < input[l].DataCount; d_i++) { //determine which Units are on for bias
		// 	for (int i = 0; i < bottom->NeuronCount - 1; i++) {
		// 		biasdata->DataInput[d_i][i] = Binary(input[l].DataInput[d_i][i]);
		//	}
		// }
		top->Neurons[top->NeuronCount - 1].InitBias(&input[l]); //initialise bias for visible units
		//for (int i = 0; i < bottom->NeuronCount - 1; i++) {
		//std::cout<<top->Neurons[top->NeuronCount - 1].Weights[i]<<std::endl;
		//}


		//Initialisieren des Netztes mit Eingabedaten vor dem Gibbs Sampling
		for (int e = 0; e < Epochs; e++) { //jede Maschine über Epochenazahl tranieren
			//std::cout << "Layer:" << l << " Epoche:" << e << " Datensatz: " << 0 << "\r" << std::flush;
			//speichern für modell statistik allokieren
			for ( int d_i = 0; d_i < input[l].DataCount; d_i++) {
				statisticsdatav[d_i] = new double[bottom->NeuronCount];
				statisticsdatah[d_i] = new double[top->NeuronCount];
				statisticsdatasm[d_i] = new double[container.OutputCount];
				statisticsmodelv[d_i] = new double[bottom->NeuronCount];
				statisticsmodelh[d_i] = new double[top->NeuronCount];
				statisticsmodelsm[d_i] = new double[container.OutputCount];
			}
			int d_i = 0;
			int d_end = 0;
			int d_start;
			for (int b = 0; b < batches; b++) {
				d_start = b * BatchSize;
				if (b == batches - 2 && rest != 0)
					BatchSize = rest;
				d_end = d_i + BatchSize;
				for (int i = 0; i < bottom->NeuronCount - 1; i++) {
					statisticsdatav[d_i][i] = 0;
					statisticsmodelv[d_i][i] = 0;
				}
				for (int j = 0; j < top->NeuronCount - 1; j++) {
					statisticsdatah[d_i][j] = 0;
					statisticsmodelh[d_i][j] = 0;
				}
				for (int i = 0; i < network->SoftmaxGroup; i++) {
					statisticsdatasm[d_i][i] = 0;
					statisticsmodelsm[d_i][i] = 0;
				}


				for (d_i=d_start; d_i < d_end; d_i++) { //Alle Trainingsdaten für jede Maschine
					for (int j = 0; j < bottom->NeuronCount - 1; ++j) { //Eingabedaten setzen
						bottom->Neurons[j].p = input[l].DataInput[d_i][j]; //Interpret data as probability to turn on
						bottom->Neurons[j].Output = Binary(bottom->Neurons[j].p);
					}
					if (network->SoftmaxGroup != 0 && l == layersend - 1) {
						for (int j = 0; j < network->SoftmaxGroup; j++) {
							network->Softmax.Neurons[j].Output = container.DataOutput[d_i][j];
							statisticsdatasm[d_i][j] = container.DataOutput[d_i][j];
						}
					}

					//std::cout << "Layer:" << l << " Epoche:" << e << " Datensatz: " << d_i << "\r" << std::flush;
					//Versteckte Neuronen berechnen
					if (network->SoftmaxGroup != 0 && l == layersend - 1) {
						UpdateHiddenUnits(network->SoftmaxGroup);
					} else
						UpdateHiddenUnits(0);
					//sammle Statistikdaten von Eingabedaten v*h
					//sammle Statistik für v_data
					for (int i = 0; i < bottom->NeuronCount - 1; i++) {
						statisticsdatav[d_i][i] = bottom->Neurons[i].p;

					}


					for (int j = 0; j < top->NeuronCount - 1; j++) {
						statisticsdatah[d_i][j] = top->Neurons[j].p;

					}

					//Gibbs Sampling
					if (network->SoftmaxGroup != 0 && l == layersend - 1)
						GibbsSampling(gibbssteps, network->SoftmaxGroup);
					else
						GibbsSampling(gibbssteps, 0);
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
					if (network->SoftmaxGroup != 0 && l == layersend - 1) {
						for (int s = 0; s < network->SoftmaxGroup; s++) {
							statisticsmodelsm[d_i][s] = network->Softmax.Neurons[s].p;
						}
					}
				} //Ende Batch
				double edata = 0.0, emodel = 0.0;

				for (int i = 0; i < bottom->NeuronCount - 1; i++) {

					for ( int j = 0; j < top->NeuronCount - 1; j++) {
						edata = 0.0;
						emodel = 0.0;
						for ( d_i=d_start; d_i < d_end; d_i++) {

							emodel += statisticsmodelv[d_i][i] * statisticsmodelh[d_i][j];
							edata += statisticsdatav[d_i][i] * statisticsdatah[d_i][j];
						}
						emodel /= BatchSize;
						edata /= BatchSize;
						//Gewichte anpassen

						//std::cout << edata << " " << emodel << std::endl;
						*bottom->Neurons[i].ForwardWeights[j] += learnRate * (edata - emodel);
					}

				}
				if (network->SoftmaxGroup != 0 && l == layersend - 1) {
					for ( int s = 0; s < network->SoftmaxGroup; s++) {
						for (int j = 0; j < top->NeuronCount - 1; j++) {
							edata = 0.0;
							emodel = 0.0;
							for (d_i=d_start; d_i < d_end; d_i++) {
								emodel += statisticsmodelsm[d_i][s] * statisticsmodelh[d_i][j];
								edata += statisticsdatasm[d_i][s] * statisticsdatah[d_i][j];
							}
							emodel /= BatchSize;
							edata /= BatchSize;
							*network->Softmax.Neurons[s].ForwardWeights[j] += learnRate * (edata - emodel);
						}
					}
				}


				//bias anpassen für visible units
				for ( int i = 0; i < bottom->NeuronCount - 1; i++) {
					emodel = 0.0;
					edata = 0.0;
					for ( d_i=d_start; d_i < d_end; d_i++) {
						emodel += statisticsmodelv[d_i][i];
						edata += statisticsdatav[d_i][i];
					}
					emodel /= BatchSize;
					edata /= BatchSize;

					top->Neurons[top->NeuronCount - 1].Weights[i] += learnRate * (edata - emodel);
				}
				//bias anpassen für hidden units
				for ( int j = 0; j < top->NeuronCount - 1; j++) {
					emodel = 0.0;
					edata = 0.0;
					for (d_i=d_start; d_i < d_end; d_i++) {
						emodel += statisticsmodelh[d_i][j];
						edata += statisticsdatah[d_i][j];
					}
					emodel /= BatchSize;
					edata /= BatchSize;
					*bottom->Neurons[bottom->NeuronCount - 1].ForwardWeights[j] += learnRate * (edata - emodel);

				}

				
			} // Ende Batches




			delete biasdata;
			biasdata = nullptr;

			for ( int d_i = 0; d_i < input[l].DataCount; d_i++) {
				delete [] statisticsdatav[d_i];
				delete [] statisticsmodelv[d_i];
				delete [] statisticsdatah[d_i];
				delete [] statisticsmodelh[d_i];
				delete [] statisticsdatasm[d_i];
				delete [] statisticsmodelsm[d_i];

			}

		}
		for ( int d_i = 0; d_i < input[l].DataCount; d_i++) {
			for (int j = 0; j < bottom->NeuronCount - 1; ++j) { //Eingabedaten setzen
				bottom->Neurons[j].p = input[l].DataInput[d_i][j]; //Interpret data as probability to turn on
				bottom->Neurons[j].Output = Binary(input[l].DataInput[d_i][j]);
			}
			//Versteckte Neuronen berechnen
			UpdateHiddenUnits(0);
			for ( int j = 0; j < top->NeuronCount - 1; j++) {
				input[l + 1].DataInput[d_i][j] = top->Neurons[j].p;
			}
			if (l + 1 == network->LayersCount - 2) {
				for ( int j = 0; j < top->NeuronCount - 1; j++) {
					input[l + 1].DataInput[d_i][j] = input[l + 1].DataInput[d_i][j];
				}
			}


		}

		delete [] statisticsdatav;
		delete [] statisticsmodelv;
		delete [] statisticsdatah;
		delete [] statisticsmodelh;
		delete [] statisticsdatasm;
		delete [] statisticsmodelsm;
	}
	if (network->SoftmaxGroup == 0) {
		int layers[2] = {network->Layers[network -> LayersCount - 2].NeuronCount - 1, network->Layers[network -> LayersCount - 1].NeuronCount - 1};
		NeuralNetwork lastlayer(layers, 2, static_cast<NNTLib::WeightInitEnum>(1), static_cast<NNTLib::FunctionEnum>(1), network->LastLayerFunction);
		Backpropagation backprop(lastlayer);
		for ( int d_i = 0; d_i < container.DataCount; d_i++)
			for (int i = 0; i < container.OutputCount; i++) {
				input[network->LayersCount - 2].DataOutput[d_i][i] = container.DataOutput[d_i][i];
			}
		backprop.Train(input[network->LayersCount - 2], learnRate, 500, 0, BatchSize, 0, 0);

		for (int i = 0; i < network->Layers[network->LayersCount - 2].NeuronCount; i++) {
			for (int j = 0; j < network->Layers[network->LayersCount - 1].NeuronCount - 1; j++) {
				//std::cout << lastlayer.Layers[0].Neurons[j].Weights[i] << std::endl;
				*network->Layers[network->LayersCount - 2].Neurons[i].ForwardWeights[j] = lastlayer.Layers[0].Neurons[j].Weights[i];
			}
		}
	}


	delete [] input;
	runtime=GetTimeMs64() - time64;

}

/**
 * @brief constructor
 * @details constructor
 * 
 * @param net network to train
 */
ContrastiveDivergence::ContrastiveDivergence(DeepBeliefNet & net) {
	this->network = &net;
	generator.seed(time(NULL));

}

/**
 * @brief destructor
 * @details destrucotr
 */
ContrastiveDivergence::~ContrastiveDivergence() {};


}
