#include "Layer.h"
namespace NNTLib
{
	/// <summary>
	/// Initializes this instance.
	/// </summary>
	void Layer::init()
	{
		InputValuesCount=0;
		InputValuesCountWithBias=0;
		NeuronCount=0;
		Neurons = nullptr;
		InputValues= nullptr;
		SumDeltaErrWeights= nullptr;
	}

	/// <summary>
	/// Frees the memory.
	/// </summary>
	void Layer::freeMem()
	{
		
		delete [] Neurons;
		delete [] InputValues;
		delete [] SumDeltaErrWeights;
	}

	/// <summary>
	/// Copies the specified that.
	/// </summary>
	/// <param name="that">The that.</param>
	void Layer::copy(const Layer &that)
	{
		this->InputValuesCount = that.InputValuesCount;
		this->InputValuesCountWithBias = that.InputValuesCountWithBias;
		this->NeuronCount = that.NeuronCount;

		this->Neurons = new Neuron[this->NeuronCount];
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

	/// <summary>
	/// Initializes a new instance of the <see cref="Layer"/> class.
	/// </summary>
	Layer::Layer()
	{
		init();
	}

	/// <summary>
	/// Finalizes an instance of the <see cref="Layer"/> class.
	/// </summary>
	Layer::~Layer()
	{
		freeMem();
	}

	/// <summary>
	/// Initializes a new instance of the <see cref="Layer"/> class.
	/// </summary>
	/// <param name="that">The that.</param>
	Layer::Layer(const Layer &that)
	{
		init();
		copy(that);
	}

	/// <summary>
	/// Operator=s the specified that.
	/// </summary>
	/// <param name="that">The that.</param>
	/// <returns></returns>
	Layer& Layer::operator= (const Layer &that)
	{
		if (&that != this) {
			freeMem();
			init();
			copy(that);
		}
		return *this;
	}

	/// <summary>
	/// Initializes the specified inputsize.
	/// </summary>
	/// <param name="inputsize">The inputsize.</param>
	/// <param name="neuronCount">The neuron count.</param>
	void Layer::Init(int inputsize, int neuronCount)
	{
		NeuronCount=neuronCount;
		InputValuesCount=inputsize;

		Neurons=new Neuron[NeuronCount];
		InputValues=new double[InputValuesCount]();
		SumDeltaErrWeights = new double[inputsize]();
		InputValuesCountWithBias= InputValuesCount+1;
		for(int i=0;i<NeuronCount;++i)
		{
			Neurons[i].Init(InputValuesCountWithBias);
		}
	}
}