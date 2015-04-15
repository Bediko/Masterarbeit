#ifndef GeneticConfig_H
#define GeneticConfig_H

#include <iostream>
#include <sstream>
#include <fstream>

#include "ConfigBase.h"

class GeneticAlgorithmConfig : public ConfigBase
{
private:
public:

	GeneticAlgorithmConfig();

	/// <summary>
	/// The error threshold
	/// </summary>
	double ErrorThreshold;
	/// <summary>
	/// The maximum loop count
	/// </summary>
	int MaxLoopCount;
	/// <summary>
	/// The mutate type
	/// </summary>
	NNTLib::MutateEnum MutateType;
	/// <summary>
	/// The cross type
	/// </summary>
	NNTLib::CrossoverEnum CrossType;
	/// <summary>
	/// The roulette type
	/// </summary>
	NNTLib::RouletteEnum RouletteType;
	/// <summary>
	/// The population size
	/// </summary>
	int PopulationSize;
	/// <summary>
	/// The eltism count
	/// </summary>
	int EltismCount;
	/// <summary>
	/// The mutation probability
	/// </summary>
	double MutationProbability;
	/// <summary>
	/// The crossover probability
	/// </summary>
	double CrossoverProbability;
	/// <summary>
	/// The mutate node count
	/// </summary>
	int MutateNodeCount;
	void PrintData();
	bool IsConfigValid();
protected:
	void HandleNameValue(std::string name,std::string value);
};
#endif
