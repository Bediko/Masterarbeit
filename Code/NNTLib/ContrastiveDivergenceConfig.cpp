#include "ContrastiveDivergenceConfig.h"

/// <summary>
/// Initializes a new instance of the <see cref="ContrastiveDivergenceConfig" /> class.
/// </summary>
ContrastiveDivergenceConfig::ContrastiveDivergenceConfig():GibbsSteps(0),BatchSize(0),LearnRate(0.0), Epochs(0)
{}

/// <summary>
/// Handles the name value.
/// </summary>
/// <param name="name">The name.</param>
/// <param name="value">The value.</param>
void ContrastiveDivergenceConfig::HandleNameValue(std::string name,std::string value)
{
	if(name == "GibbsSteps")
		GibbsSteps= atoi(value.c_str());
	
	else if(name == "BatchSize")
		BatchSize = atoi(value.c_str());
	else if(name == "LearnRate")
		LearnRate = atof(value.c_str());
	else if (name== "Epochs")
		Epochs = atoi(value.c_str());
}

/// <summary>
/// Prints the data.
/// </summary>
void ContrastiveDivergenceConfig::PrintData()
{
	std::cout <<"ContrastiveDivergenceConfig:"<<std::endl;
	std::cout <<"GibbsSteps = "<<GibbsSteps<<std::endl;
	std::cout <<"BatchSize = "<<BatchSize<<std::endl;
	std::cout <<"LearnRate = "<<LearnRate<<std::endl;
	std::cout <<"Epochs = "<<Epochs<<std::endl;
}

/// <summary>
/// Determines whether [is configuration valid].
/// </summary>
/// <returns></returns>
bool ContrastiveDivergenceConfig::IsConfigValid()
{
	return true;
}