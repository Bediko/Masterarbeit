#include "DBNNeuron.h"
#include <iostream>

namespace NNTLib {
/**
 * @brief Initialises Neuron with default Values
 * @details  Sets everything to 0 and the bias to 1
 */
void DBNNeuron::init() {
	WeightCount = 0;
	ForwardWeightCount = 0;
	ForwardWeights = nullptr;
	Output = 0;
	Weights = nullptr;
	LastDeltaWeights = nullptr;
	DeltaWeights = nullptr;
	Bias = 1;
	p = 0;
}
/**
 * @brief standard constructor
 */
DBNNeuron::DBNNeuron() {
	init();
}
/**
 * @brief Frees memory
 * @details deletes Forwardweights
 */
void DBNNeuron::freeMem() {
	//delete [] Weights;
	//delete [] LastDeltaWeights;
	//delete [] DeltaWeights;
	delete [] ForwardWeights;
}
/**
 * @brief destructor
 */
DBNNeuron::~DBNNeuron() {
	freeMem();
}

/**
 * @brief Copy Constructor
 * @param that Neuron to copy
 */
DBNNeuron::DBNNeuron(const DBNNeuron &that) {
	init();
	copy(that);
}


/**
 * @brief Copies Neuron
 * @details Copies Neuron into a new one
 *
 * @param that Neuron to copy
 */
void DBNNeuron::copy(const DBNNeuron &that) {
	this->Bias = that.Bias;
	this->WeightCount = that.WeightCount;
	this->Output = that.Output;
	this->ForwardWeightCount = that.ForwardWeightCount;

	this->Weights = new double[WeightCount];
	this->DeltaWeights = new double[WeightCount];
	this->LastDeltaWeights = new double[WeightCount];
	this->ForwardWeights = new double*[ForwardWeightCount];

	for (int i = 0; i < WeightCount; i++) {
		this->Weights[i] = that.Weights[i];
		this->DeltaWeights[i] = that.DeltaWeights[i];
		this->LastDeltaWeights[i] = that.LastDeltaWeights[i];
	}
	for (int i = 0; i < ForwardWeightCount; i++)
		this->ForwardWeights[i] = that.ForwardWeights[i];
}
/**
 * @brief Overloads =
 * @param that Neuron to copy
 * @return new Neuron with copied values
 */
DBNNeuron& DBNNeuron::operator= (const DBNNeuron &that) {
	if (&that != this) {
		freeMem();
		init();
		copy(that);
	}
	return *this;
}

/**
 * @brief Initialises Biasneuron
 * @details Calculates $\log[p_i/(1-p_i)]$ with $p_i$ as the average propability
 * that a neuron turns on vor the visible layer. Hidden Layer gets Bias set to 0.
 * 
 * @param container Datacontainer with the probabilites for every training example
 */
void DBNNeuron::InitBias(const DataContainer* container) {
	if (container == NULL) {
		for (int i = 0; i < ForwardWeightCount; i++)
			*ForwardWeights[i] = 0;
		return;
	}
	double *pi = new double[container->InputCount];
	for (int i = 0; i < container->InputCount; i++) {
		pi[i] = 0;
	}
	for (int d_i = 0; d_i < container->DataCount; d_i++) {
		for (int i = 0; i < container->InputCount; i++) {
			//if (container->DataInput[d_i][i] > 0.0) {
			pi[i] += container->DataInput[d_i][i];
			//}

		}
	}
	for (int i = 0; i < container->InputCount; i++) {
		double count = (double)container->DataCount;
		pi[i] = pi[i] / count;
		pi[i] = log(pi[i] / (1 - pi[i]));
		Weights[i] = pi[i];
	}
	delete [] pi;
}

/**
 * @brief Initialises Neuron
 * @details Allocates weights, deltaeights and lastdeltaweights with weightcount and sets weightcount
 * 
 * @param weightCount Number of Backward weights on the neuron
 */
void DBNNeuron::Init(int weightCount) {
	WeightCount = weightCount;
	Weights = new double[WeightCount]();
	DeltaWeights = new double[WeightCount]();
	LastDeltaWeights = new double[WeightCount]();
}
}