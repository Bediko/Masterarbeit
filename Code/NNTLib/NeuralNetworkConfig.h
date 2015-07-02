#ifndef NeuralNetworkTrainConfigBase_H
#define NeuralNetworkTrainConfigBase_H
#include <iostream>
#include <sstream>
#include <fstream>

//#include "Functions.h"
#include "ConfigBase.h"

class NeuralNetworkConfig : public ConfigBase
{
private:
	void init();
	void freeMem();
	void copy(const NeuralNetworkConfig &that);
public:

	/// <summary>
	/// The layer count
	/// </summary>
	int LayerCount;
	/// <summary>
	/// The function type
	/// </summary>
	NNTLib::FunctionEnum FunctionType;
	NNTLib::FunctionEnum LastLayerFunction;
	/// <summary>
	/// The weight initialize type
	/// </summary>
	NNTLib::WeightInitEnum WeightInitType;
	/// <summary>
	/// The layer neuron count
	/// </summary>
	int * LayerNeuronCount;

	NeuralNetworkConfig();
	~ NeuralNetworkConfig();

	NeuralNetworkConfig(const NeuralNetworkConfig &that);

	NeuralNetworkConfig& operator= (const NeuralNetworkConfig &that);
	void PrintData();
	bool IsConfigValid();

protected:
	void HandleNameValue(std::string name,std::string value);
};

#endif
