#include "NeuralNetworkConfig.h"

/// <summary>
/// Initializes this instance.
/// </summary>
void NeuralNetworkConfig::init()
{
	LayerCount=0;
	FunctionType=NNTLib::FunctionEnum::LINEAR;
	WeightInitType=NNTLib::WeightInitEnum::NONE;
	LayerNeuronCount=nullptr;
}
/// <summary>
/// Frees the memory.
/// </summary>
void NeuralNetworkConfig::freeMem()
{
	if(LayerNeuronCount)
	{
		delete [] LayerNeuronCount;
	}
}
/// <summary>
/// Copies the specified that.
/// </summary>
/// <param name="that">The that.</param>
void NeuralNetworkConfig::copy(const NeuralNetworkConfig &that)
{
	LayerCount=that.LayerCount;
	FunctionType=that.FunctionType;
	WeightInitType=that.WeightInitType;

	this->LayerNeuronCount = new int[LayerCount];

	for(int i=0;i<LayerCount;i++)
		this->LayerNeuronCount[i] = that.LayerNeuronCount[i];
}

/// <summary>
/// Initializes a new instance of the <see cref="NeuralNetworkConfig" /> class.
/// </summary>
NeuralNetworkConfig::NeuralNetworkConfig()
{
	init();
}

/// <summary>
/// Finalizes an instance of the <see cref="NeuralNetworkConfig" /> class.
/// </summary>
NeuralNetworkConfig::~NeuralNetworkConfig()
{
	freeMem();
}

/// <summary>
/// Initializes a new instance of the <see cref="NeuralNetworkConfig"/> class.
/// </summary>
/// <param name="that">The that.</param>
NeuralNetworkConfig::NeuralNetworkConfig(const NeuralNetworkConfig &that)
{
	init();
	copy(that);
}

/// <summary>
/// Operator=s the specified that.
/// </summary>
/// <param name="that">The that.</param>
/// <returns></returns>
NeuralNetworkConfig& NeuralNetworkConfig::operator= (const NeuralNetworkConfig &that)
{
	if (&that != this) {
		freeMem();
		init();
		copy(that);
	}
	return *this;
}

/// <summary>
/// Handles the name value.
/// </summary>
/// <param name="name">The name.</param>
/// <param name="value">The value.</param>
void NeuralNetworkConfig::HandleNameValue(std::string name,std::string value)
{
	if(name == "FunctionType")
		FunctionType = static_cast<NNTLib::FunctionEnum>(atoi(value.c_str()));
	else if(name == "WeightInitType")
		WeightInitType = static_cast<NNTLib::WeightInitEnum>(atoi(value.c_str()));
	else if(name == "LayerCount")
		LayerCount = atoi(value.c_str());
	else if(name == "LayerNeuronCount")
	{
		LayerNeuronCount = new int[LayerCount];
		std::string segment;
		std::stringstream streamValueInput(value);

		for(int i=0;i<LayerCount;i++)
		{
			getline(streamValueInput, segment,',');
			LayerNeuronCount[i]=atoi(segment.c_str());
		}
	}
}

/// <summary>
/// Prints the data.
/// </summary>
void NeuralNetworkConfig::PrintData()
{
	std::cout <<"NeuralNetworkConfig:"<<std::endl;
	std::cout <<"FunctionType = "<<GetStringValue(FunctionType)<<std::endl;
	std::cout <<"WeightInitType = "<< GetStringValue(WeightInitType)<<std::endl;
	std::cout <<"LayerCount = "<<LayerCount<<std::endl;
	std::cout <<"LayerNeuronCount = ";
	for(int i=0;i<LayerCount;i++)
	{
		std::cout <<LayerNeuronCount[i];
		if(i != LayerCount-1)
			std::cout <<",";
	}
	std::cout << std::endl;
}

/// <summary>
/// Determines whether [is configuration valid].
/// </summary>
/// <returns></returns>
bool NeuralNetworkConfig::IsConfigValid()
{
	return true;
}