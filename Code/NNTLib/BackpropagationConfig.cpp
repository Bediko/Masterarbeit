#include "BackpropagationConfig.h"

/// <summary>
/// Initializes a new instance of the <see cref="BackpropagationConfig" /> class.
/// </summary>
BackpropagationConfig::BackpropagationConfig():ErrorThreshold(0),MaxLoopCount(0),BatchSize(0),Alpha(0),Momentum(0),DecayRate(0)
{}

/// <summary>
/// Handles the name value.
/// </summary>
/// <param name="name">The name.</param>
/// <param name="value">The value.</param>
void BackpropagationConfig::HandleNameValue(std::string name,std::string value)
{
	if(name == "MaxLoopCount")
		MaxLoopCount= atoi(value.c_str());
	else if(name == "ErrorThreshold")
		ErrorThreshold = atof(value.c_str());
	else if(name == "BatchSize")
		BatchSize = atoi(value.c_str());
	else if(name == "Alpha")
		Alpha = atof(value.c_str());
	else if(name == "Momentum")
		Momentum = atof(value.c_str());
	else if(name == "DecayRate")
		DecayRate = atof(value.c_str());
}

/// <summary>
/// Prints the data.
/// </summary>
void BackpropagationConfig::PrintData()
{
	std::cout <<"BackpropagationConfig:"<<std::endl;
	std::cout <<"MaxLoopCount = "<<MaxLoopCount<<std::endl;
	std::cout <<"ErrorThreshold = "<<ErrorThreshold<<std::endl;
	std::cout <<"BatchSize = "<<BatchSize<<std::endl;
	std::cout <<"Alpha = "<<Alpha<<std::endl;
	std::cout <<"MaxLoopCount = "<<MaxLoopCount<<std::endl;
	std::cout <<"Momentum = "<<Momentum<<std::endl;
	std::cout <<"DecayRate = "<<DecayRate<<std::endl;
}

/// <summary>
/// Determines whether [is configuration valid].
/// </summary>
/// <returns></returns>
bool BackpropagationConfig::IsConfigValid()
{
	return true;
}