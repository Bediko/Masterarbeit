#include "Neuron.h"
#include <iostream>

namespace NNTLib
{
	/// <summary>
	/// Initializes this instance.
	/// </summary>
	void Neuron::init()
	{
		WeightCount=0;
		ForwardWeightCount=0;
		ForwardWeights=nullptr;
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
		this->ForwardWeightCount = that.ForwardWeightCount;

		this->Weights=new double[WeightCount];
		this->DeltaWeights=new double[WeightCount];
		this->LastDeltaWeights=new double[WeightCount];
		this->ForwardWeights = new double*[ForwardWeightCount];

		for(int i=0;i<WeightCount;i++)
		{
			this->Weights[i]=that.Weights[i];
			this->DeltaWeights[i]=that.DeltaWeights[i];
			this->LastDeltaWeights[i]=that.LastDeltaWeights[i];
		}
		for(int i=0;i<ForwardWeightCount;i++)
			this->ForwardWeights[i]=that.ForwardWeights[i];
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
	void Neuron::InitBias(const DataContainer* container){
		if(container==NULL){
			for(int i=0;i<ForwardWeightCount;i++)
				*ForwardWeights[i]=0;
			return;
		}

		double *pi= new double[container->InputCount];
		for(int i=0;i<container->InputCount;i++){
			pi[i]=0;
		}
		for (int d_i=0; d_i<container->DataCount;d_i++){
			for (int i=0;i<container->InputCount;i++){
				if(container->DataInput[d_i][i]>0.0){
					pi[i]+=1.0;
				}

			}
		}
		

		for(int i=0;i<container->InputCount;i++){
			double count=(double)container->DataCount;
			pi[i]=pi[i]/count;

			pi[i]= log(pi[i]/(1-pi[i]));

			Weights[i]=pi[i];
			
		}


	}
}