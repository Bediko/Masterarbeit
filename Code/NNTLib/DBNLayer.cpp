#include "DBNLayer.h"
#include "DBNNeuron.h"

namespace NNTLib
{
	void DBNLayer::Init(int inputsize, int neuronCount)
	{
		NeuronCount=neuronCount;
		InputValuesCount=inputsize;

		Neurons=new Neuron[NeuronCount];
		InputValues=new double[InputValuesCount]();
		SumDeltaErrWeights = new double[inputsize]();
		if(inputsize!=0)
			InputValuesCountWithBias= InputValuesCount+1;

		for(int i=0;i<NeuronCount-1;++i)
		{
			Neurons[i].Init(InputValuesCountWithBias);
		}
		Neurons[NeuronCount-1].Init(InputValuesCount);//bias hat keine Eingabegewichte
	}
	void DBNLayer::Forwardweightsinit(int Neuronsdown, Layer* Layerup, int dbn)
	{
		for (int i=0;i<=Neuronsdown;i++){
			Neurons[i].ForwardWeightCount=Layerup->NeuronCount-1;
			Neurons[i].ForwardWeights= new double*[Neurons[i].ForwardWeightCount];
			for (int j=0; j<Layerup->NeuronCount-1;j++){ //-1 da keine Gewichte zum Bias
				Neurons[i].ForwardWeights[j]=&Layerup->Neurons[j].Weights[i];
			}
		}

	}
	void DBNLayer::copy(const Layer &that)
	{
		this->InputValuesCount = that.InputValuesCount;
		this->InputValuesCountWithBias = that.InputValuesCountWithBias;
		this->NeuronCount = that.NeuronCount;

		this->Neurons = new DBNNeuron[this->NeuronCount];
		this->InputValues = new double[this->InputValuesCount];
		this->SumDeltaErrWeights = new double[this->InputValuesCount];

		for(int i=0;i<this->NeuronCount;i++)
			this->Neurons[i] = that.Neurons[i];

		for(int i=0;i<InputValuesCount;i++)
		{
			this->InputValues[i]=that.InputValues[i];
			this->SumDeltaErrWeights[i]=that.SumDeltaErrWeights[i];
		}
	}
}