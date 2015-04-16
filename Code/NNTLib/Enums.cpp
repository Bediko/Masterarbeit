#include "Enums.h"

namespace NNTLib
{
	//Mappe enums mit strings
	ENUM_MAP(FunctionEnum,"LINEAR,LOGISTIC,TANH,LECUN_TANH")
		ENUM_MAP(MutateEnum,"NONE,UNBIASED_MUTATA_WEIGHTS, BIAS_ED_MUTATE_WEIGHTS,MUTATE_NODES")
		ENUM_MAP(CrossoverEnum,"NONE,CROSSOVER_WEIGHTS,CROSSOVER_NODES,CROSSOVER_ONEPOINT,CROSSOVER_TWOPOINT")
		ENUM_MAP(RouletteEnum,"FITTNESSBASED,INDEXBASED")
		ENUM_MAP(WeightInitEnum,"NONE,UNIFORM,LECUN,UNIFORM5,NORMAL5,DEBUGONE,NORMAL0")

		//v entspricht index im Enum
		//tmp entspricht mit ',' seperierte string liste der enum namen
		std::string GetEnumName(unsigned int v, std::string tmp)
	{
		std::vector<std::string> elems;
		std::stringstream ss(tmp);
		std::string item;

		while(std::getline(ss, item, ','))
		{
			elems.push_back(item);
		}

		if(v >= elems.size())
		{
			return "UNKNOWN ENUM";
		}

		return elems[v];
	}
}