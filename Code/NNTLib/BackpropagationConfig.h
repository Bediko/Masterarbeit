#ifndef BackpropagationConfig_H
#define BackpropagationConfig_H

#include <iostream>
#include <sstream>
#include <fstream>

#include "ConfigBase.h"

class BackpropagationConfig : public ConfigBase
{
private:

public:
	BackpropagationConfig();

	/// <summary>
	/// The error threshold
	/// </summary>
	double ErrorThreshold;
	/// <summary>
	/// The maximum loop count
	/// </summary>
	int MaxLoopCount;
	/// <summary>
	/// The batch size
	/// </summary>
	int BatchSize;
	/// <summary>
	/// The alpha
	/// </summary>
	double Alpha;
	/// <summary>
	/// The momentum
	/// </summary>
	double Momentum;
	/// <summary>
	/// The decay rate
	/// </summary>
	double DecayRate;
	void PrintData();
	bool IsConfigValid();
protected:
	void HandleNameValue(std::string name,std::string value);
};

#endif