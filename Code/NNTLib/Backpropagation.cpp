#include "Backpropagation.h"

namespace NNTLib
{
	/// <summary>
	/// Initializes a new instance of the <see cref="Backpropagation" /> class.
	/// </summary>
	/// <param name="net">The net.</param>
	Backpropagation::Backpropagation(NeuralNetwork &net)
	{
		this->network = &net;
	}

	/// <summary>
	/// Finalizes an instance of the <see cref="Backpropagation" /> class.
	/// </summary>
	Backpropagation::~Backpropagation()
	{
	}

	/// <summary>
	/// Trains the specified network-&gt;
	/// </summary>
	/// <param name="container">The container.</param>
	/// <param name="learnRate">The learnRate.</param>
	/// <param name="maxLoopCount">The maxLoopCount.</param>
	/// <param name="momentum">The momentum.</param>
	/// <param name="errorThreshold">The errorThreshold.</param>
	/// <param name="decayRate">The decay rate.</param>
	void Backpropagation::trainIncremental(const DataContainer &container,const double learnRate,const int maxLoopCount,const double momentum,const double errorThreshold,const double decayRate)
	{
		//lokale variablen zur verbesserung der performance
		//Zugriffszeit auf weights[k] besser als auf network->Layers[i]->layer->Neurons[j]->weights[k];
		int i,j,k,l;
		double error;
		//double *deltaWeights;
		Layer* layer;
		Neuron *neuron;
		double *weights;
		double *lastDeltaWeights;
		double *inputVector;
		double *sumDeltaErrWeights = nullptr;
		double *sumDeltaErrWeightsNext;
		double error_x_learnrate;
		//double deltaWeight;
		int inputVectorCount;

		double decayedLearnRate = learnRate;
		unsigned long long time64= GetTimeMs64();

		for(l=0;l<maxLoopCount;++l)
		{
			double squareErrorSum=0;
			//network->MeanSquareError = 0;
			//Anfang Backpropagation Schritt
			for(int d_i=0;d_i<container.DataCount;++d_i)
			{
				//Daten durch das netz "propagieren"
				network->Propagate(container.DataInput[d_i]);
				//Fehler des letzten Layers berechnen, Gewichte Anpassen und Gewichtetete Fehler für Layer l-1 berechnen
				//pointer zugriffe zu minimieren
				//int lastLayerIndex = network->LayersCount - 1;
				for(i=network->LayersCount - 1;i>=0;i--)//zurück propagieren fangen beim letzten Layer an
				{
					layer = &network->Layers[i];
					inputVector = layer->InputValues;
					sumDeltaErrWeightsNext = layer->SumDeltaErrWeights;
					inputVectorCount = layer->InputValuesCount;

					for(j=0;j<layer->NeuronCount;++j)
					{
						neuron = &layer->Neurons[j];
						weights = neuron->Weights;
						lastDeltaWeights = neuron->LastDeltaWeights;
						//deltaWeights = neuron->DeltaWeights;

						if(i==network->LayersCount - 1)//Letzter Layer
						{
							//Fehler berechnen , error = f_act'(net) * (t_i - o_j)
							error=  ActivationFunctionDerivate(network->LastLayerFunction,neuron->Output) * (container.DataOutput[d_i][j] - neuron->Output);
							//std::cout<<ActivationFunctionDerivate(network->LastLayerFunction,neuron->Output)<<std::endl;
							//std::cout<<neuron->Output<<std::endl;
							//Quadratischen Fehler berechnen und aufsummineren
							squareErrorSum+=(container.DataOutput[d_i][j] - neuron->Output) * (container.DataOutput[d_i][j] - neuron->Output) ;
						}
						else
						{
							//Fehler berechnen = error = f_act'(net) * Sum( Fehler vorherigen Layers * Gewicht)
							error= ActivationFunctionDerivate(network->FunctionType,neuron->Output) * sumDeltaErrWeights[j];
							sumDeltaErrWeights[j] = 0; // für nächsten Trainingssatz auf 0 setzen
						}

						//Fehler mit lernrate multipliziert
						error_x_learnrate = decayedLearnRate * error;
						//ONLINE SPEZIFISCHER PART BEGIN
						if(i!=0)
						{
							for(k=0;k<inputVectorCount;++k)
							{
								//Fehler Sum( Fehler vorherigen Layers * Gewicht) für die Neuronen des nächsten Layers l-1 berechnen
								sumDeltaErrWeightsNext[k] += weights[k] * error;
								//gewichtsänderung speichern für nächste iteration
								lastDeltaWeights[k]=error_x_learnrate * inputVector[k] + lastDeltaWeights[k] * momentum;
								//Gewicht anpassen
								weights[k]+=lastDeltaWeights[k];
							}
						}
						else
						{
							//Fehler Sum( Fehler vorherigen Layers * Gewicht) für die Neuronen des nächsten Layers l-1 nicht nötig da wir uns im letzten (mit Neuronen) befinden
							//Gewichte aktualisieren
							for(k=0;k!=inputVectorCount;++k)
							{
								lastDeltaWeights[k]= (error_x_learnrate * inputVector[k]) + (lastDeltaWeights[k] * momentum);
								weights[k]+=lastDeltaWeights[k] ;
							}
						}
						//Schwellenwert berechnen
						lastDeltaWeights[inputVectorCount]=  error_x_learnrate /** neuron->Bias*/ + lastDeltaWeights[inputVectorCount] * momentum;
						weights[inputVectorCount]+=lastDeltaWeights[inputVectorCount];

						//ONLINE SPEZIFISCHER PART END
					}
					sumDeltaErrWeights = layer->SumDeltaErrWeights;
				}
			}
			//Ende Backpropagation Schritt
			//Lernrate anpassen (falls decayRate=0 ist die lernrate statisch)
			decayedLearnRate = learnRate / (1 + (l * decayRate));

			//Trainingsergebnisse setzen (MSE/Ausführungszeit in ms)
			MeasureResult[l].MeanSquareError = squareErrorSum / container.DataCount;//MSE aus SE berechnen
			MeasureResult[l].ExecuteTime =GetTimeMs64() - time64;

			//Abbruch bedinung prüfen
			if(MeasureResult[l].MeanSquareError <= errorThreshold)
			{
				break;
			}
		}
		MeasureFilledResultLenght = l;
		//mse des netzes setzen
		network->MeanSquareError = MeasureResult[l-1].MeanSquareError;
	}

	/// <summary>
	/// Trains the batch.
	/// </summary>
	/// <param name="container">The container.</param>
	/// <param name="learnRate">The learn rate.</param>
	/// <param name="maxLoopCount">The maximum loop count.</param>
	/// <param name="momentum">The momentum.</param>
	/// <param name="minibatchSize">Size of the batch.</param>
	/// <param name="errorThreshold">The error threshold.</param>
	/// <param name="decayRate">The decay rate.</param>
	void Backpropagation::trainBatch(const DataContainer &container,const double learnRate,const int maxLoopCount,const double momentum,int minibatchSize,const double errorThreshold,const double decayRate)
	{
		//lokale variablen zur verbesserung der performance
		//Zugriffszeit auf weights[k] besser als auf network->Layers[i]->layer->Neurons[j]->weights[k];
		int i,j,k,l;
		double error;
		double *deltaWeights;
		Layer* layer;
		Neuron *neuron;
		double *weights;
		double *lastDeltaWeights;
		double *inputVector;
		double *sumDeltaErrWeights = nullptr;
		double *sumDeltaErrWeightsNext;
		double error_x_learnrate;
		//double deltaWeight;
		int inputVectorCount;

		double decayedLearnRate = learnRate;
		unsigned long long time64= GetTimeMs64();

		for(l=0;l<maxLoopCount;++l)
		{
			double squareErrorSum=0;
			//network->MeanSquareError = 0;
			//Anfang Backpropagation Schritt
			for(int d_i=0;d_i<container.DataCount;++d_i)
			{
				int iteration_batch=d_i+1;
				//Daten durch das netz "propagieren"
				network->Propagate(container.DataInput[d_i]);
				//Fehler des letzten Layers berechnen, Gewichte Anpassen und Gewichtetete Fehler für Layer l-1 berechnen
				//pointer zugriffe zu minimieren
				//int lastLayerIndex = network->LayersCount - 1;
				for(i=network->LayersCount - 1;i>=0;i--)//zurück propagieren fangen beim letzten Layer an
				{
					layer = &network->Layers[i];
					inputVector = layer->InputValues;
					sumDeltaErrWeightsNext = layer->SumDeltaErrWeights;
					inputVectorCount = layer->InputValuesCount;

					for(j=0;j<layer->NeuronCount;++j)
					{
						neuron = &layer->Neurons[j];
						weights = neuron->Weights;
						lastDeltaWeights = neuron->LastDeltaWeights;
						deltaWeights = neuron->DeltaWeights;

						if(i==network->LayersCount - 1)//Letzter Layer
						{
							//Fehler berechnen , error = f_act'(net) * (t_i - o_j)
							error=  ActivationFunctionDerivate(network->FunctionType,neuron->Output) * (container.DataOutput[d_i][j] - neuron->Output);
							//Quadratischen Fehler berechnen und aufsummineren
							squareErrorSum+=(container.DataOutput[d_i][j] - neuron->Output) * (container.DataOutput[d_i][j] - neuron->Output) ;
						}
						else
						{
							//Fehler berechnen = error = f_act'(net) * Sum( Fehler vorherigen Layers * Gewicht)
							error= ActivationFunctionDerivate(network->FunctionType,neuron->Output) * sumDeltaErrWeights[j];
							sumDeltaErrWeights[j] = 0; // für nächsten Trainingssatz auf 0 setzen
						}

						//Fehler mit lernrate multipliziert
						error_x_learnrate = decayedLearnRate * error;

						//BATCH SPEZIFISCHER PART BEGIN
						if(iteration_batch % minibatchSize == 0 /*&& c != 0*/)
						{
							if(i!=0)
							{
								for(k=0;k<layer->InputValuesCount;++k)
								{
									//Fehler für die Neuronen des nächsten Layers l-1 berechnen
									sumDeltaErrWeightsNext[k] += weights[k] * error;
									//Gewichte anpassen
									deltaWeights[k]+=  error_x_learnrate * inputVector[k] + lastDeltaWeights[k] * momentum;
									weights[k]+=deltaWeights[k] ;
									lastDeltaWeights[k]=deltaWeights[k];
									deltaWeights[k]=0;
								}
							}
							else
							{
								for(k=0;k<layer->InputValuesCount;++k)
								{
									//Fehler für die Neuronen des nächsten Layers l-1 berechnung nicht nötig da wir uns im letzten Layer befinden
									//sumDeltaErrWeightsNext[k] += weights[k] * error;
									//Gewichte anpassen
									deltaWeights[k]+=  error_x_learnrate * inputVector[k] + lastDeltaWeights[k] * momentum;
									weights[k]+=deltaWeights[k] ;
									lastDeltaWeights[k]=deltaWeights[k];
									deltaWeights[k]=0;
								}
							}

							//Bias einzelnd berechnen
							deltaWeights[inputVectorCount] += /* neuron->Bias**/ error_x_learnrate  + lastDeltaWeights[inputVectorCount] * momentum;
							weights[inputVectorCount]+=deltaWeights[inputVectorCount];
							lastDeltaWeights[inputVectorCount]=deltaWeights[inputVectorCount];
							deltaWeights[inputVectorCount] =0;
						}
						else
						{
							if(i!=0)
							{
								for(k=0;k<inputVectorCount;++k)
								{
									sumDeltaErrWeightsNext[k] += weights[k] * error;
									deltaWeights[k]+=  error_x_learnrate * inputVector[k];
								}
							}
							else
							{
								for(k=0;k<inputVectorCount;++k)
								{
									//sumDeltaErrWeightsNext[k] += weights[k] * error;
									deltaWeights[k]+=  error_x_learnrate * inputVector[k];
								}
							}
							deltaWeights[inputVectorCount] += /* neuron->Bias**/ error_x_learnrate  ;
						}
						//BATCH SPEZIFISCHER PART ENDE
					}
					sumDeltaErrWeights = layer->SumDeltaErrWeights;
				}
			}
			//Ende Backpropagation Schritt
			//Lernrate anpassen (falls decayRate=0 ist die lernrate statisch)
			decayedLearnRate = learnRate / (1 + (l * decayRate));

			//Trainingsergebnisse setzen (MSE/Ausführungszeit in ms)
			MeasureResult[l].MeanSquareError = squareErrorSum / container.DataCount;//MSE aus SE berechnen
			MeasureResult[l].ExecuteTime =GetTimeMs64() - time64;

			//Abbruch bedinung prüfen
			if(MeasureResult[l].MeanSquareError <= errorThreshold)
				break;
		}
		MeasureFilledResultLenght = l;
		//mse des netzes setzen
		network->MeanSquareError = MeasureResult[l-1].MeanSquareError;
	}

	/// <summary>
	/// Trains the specified container.
	/// </summary>
	/// <param name="container">The container.</param>
	/// <param name="learnRate">The learn rate.</param>
	/// <param name="maxLoopCount">The maximum loop count.</param>
	/// <param name="momentum">The momentum.</param>
	/// <param name="minibatchSize">Size of the batch.</param>
	/// <param name="errorThreshold">The error threshold.</param>
	/// <param name="decayRate">The decay rate.</param>
	void Backpropagation::Train(const DataContainer & container,const double learnRate,const int maxLoopCount,const double momentum,int minibatchSize,const double errorThreshold,const double decayRate)
	{
		if(maxLoopCount == 0)
			return;

		initMeasureResult(maxLoopCount);

		if(minibatchSize == 1)
		{
			trainIncremental(container,learnRate,maxLoopCount,momentum,errorThreshold,decayRate);
			return;
		}
		else
		{
			if(minibatchSize > container.DataCount || minibatchSize <= 0)
				throw std::runtime_error("incrorrect minibatchsize");//vermeiden das bei zu großer batchsize einfach garnichts passiert

			trainBatch(container,learnRate,maxLoopCount,momentum,minibatchSize,errorThreshold,decayRate);
		}
	}
}