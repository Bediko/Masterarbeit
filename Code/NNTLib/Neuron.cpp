#include "Neuron.h"

namespace NNTLib
{
	/// <summary>
	/// Initializes this instance.
	/// </summary>
	void Neuron::init()
	{
		WeightCount=0;
		Output=0;
		Weights=nullptr;
		LastDeltaWeights= nullptr;
		DeltaWeights=nullptr;
		Bias=1;
	}

	/// <summary>
	/// Frees the memory.
	/// </summary>
	void Neuron::freeMem()
	{
		delete [] Weights;
		delete [] LastDeltaWeights;
		delete [] DeltaWeights;
	}
	/// <summary>
	/// Initializes a new instance of the <see cref="Neuron"/> class.
	/// </summary>
	Neuron::Neuron()
	{
		init();
	}

	/// <summary>
	/// Finalizes an instance of the <see cref="Neuron"/> class.
	/// </summary>
	Neuron::~Neuron()
	{
		freeMem();
	}

	/// <summary>
	/// Initializes a new instance of the <see cref="Neuron"/> class.
	/// </summary>
	/// <param name="that">The that.</param>
	Neuron::Neuron(const Neuron &that)
	{
		init();
		copy(that);
	}

	/// <summary>
	/// Operator=s the specified that.
	/// </summary>
	/// <param name="that">The that.</param>
	/// <returns></returns>
	Neuron& Neuron::operator= (const Neuron &that)
	{
		if (&that != this) {
			freeMem();
			init();
			copy(that);
		}
		return *this;
	}

	/// <summary>
	/// Copies the specified that.
	/// </summary>
	/// <param name="that">The that.</param>
	void Neuron::copy(const Neuron &that)
	{
		this->Bias = that.Bias;
		this->WeightCount = that.WeightCount;
		this->Output = that.Output;

		this->Weights=new double[WeightCount];
		this->DeltaWeights=new double[WeightCount];
		this->LastDeltaWeights=new double[WeightCount];

		for(int i=0;i<WeightCount;i++)
		{
			this->Weights[i]=that.Weights[i];
			this->DeltaWeights[i]=that.DeltaWeights[i];
			this->LastDeltaWeights[i]=that.LastDeltaWeights[i];
		}
	}

	/// <summary>
	/// Initializes the specified input vector count.
	/// </summary>
	/// <param name="inputVectorCount">The input vector count.</param>
	void Neuron::Init(int weightCount)
	{
		WeightCount = weightCount;

		Weights=new double[WeightCount]();
		DeltaWeights=new double[WeightCount]();
		LastDeltaWeights=new double[WeightCount]();
	}
}