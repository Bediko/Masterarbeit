#include "GeneticAlgorithmConfig.h"

/// <summary>
/// Initializes a new instance of the <see cref="GeneticAlgorithmConfig" /> class.
/// </summary>
GeneticAlgorithmConfig::GeneticAlgorithmConfig():ErrorThreshold(0),MaxLoopCount(0),MutateType(NNTLib::MutateEnum::NONE),CrossType(NNTLib::CrossoverEnum::NONE),RouletteType(NNTLib::RouletteEnum::FITTNESSBASED),PopulationSize(0),EltismCount(0),MutationProbability(0),CrossoverProbability(0),MutateNodeCount(0)
{
}

/// <summary>
/// Handles the name value.
/// </summary>
/// <param name="name">The name.</param>
/// <param name="value">The value.</param>
void GeneticAlgorithmConfig::HandleNameValue(std::string name,std::string value)
{
	if(name == "MaxLoopCount")
		MaxLoopCount= atoi(value.c_str());
	else if(name == "ErrorThreshold")
		ErrorThreshold = atof(value.c_str());
	else if(name == "MutateType")
		MutateType = static_cast<NNTLib::MutateEnum>(atoi(value.c_str()));
	else if(name == "CrossType")
		CrossType = static_cast<NNTLib::CrossoverEnum>(atoi(value.c_str()));
	else if(name == "PopulationSize")
		PopulationSize = atoi(value.c_str());
	else if(name == "RouletteType")
		RouletteType = static_cast<NNTLib::RouletteEnum>(atoi(value.c_str()));
	else if(name == "EltismCount")
		EltismCount = atoi(value.c_str());
	else if(name == "CrossoverProbability")
		CrossoverProbability = atof(value.c_str());
	else if(name == "MutationProbability")
		MutationProbability = atof(value.c_str());
	else if(name == "MutateNodeCount")
		MutateNodeCount = atoi(value.c_str());
}

/// <summary>
/// Prints the data.
/// </summary>
void GeneticAlgorithmConfig::PrintData()
{
	std::cout <<"GeneticAlgorithmConfig:"<<std::endl;
	std::cout <<"MaxLoopCount = "<<MaxLoopCount<<std::endl;
	std::cout <<"ErrorThreshold = "<<ErrorThreshold<<std::endl;
	std::cout <<"MutateType = "<<NNTLib::GetStringValue(MutateType)<<std::endl;
	std::cout <<"CrossType = "<<NNTLib::GetStringValue(CrossType)<<std::endl;
	std::cout <<"RouletteType = "<<NNTLib::GetStringValue(RouletteType)<<std::endl;
	std::cout <<"PopulationSize = "<<PopulationSize<<std::endl;
	std::cout <<"EltismCount = "<<EltismCount<<std::endl;
	std::cout <<"CrossoverProbability = "<<CrossoverProbability<<std::endl;
	std::cout <<"MutationProbability = "<<MutationProbability<<std::endl;
	std::cout <<"MutateNodeCount = "<<MutateNodeCount<<std::endl;
}

/// <summary>
/// Determines whether [is configuration valid].
/// </summary>
/// <returns></returns>
bool GeneticAlgorithmConfig::IsConfigValid()
{
	return true;
}