#ifndef ENUMS_H
#define ENUMS_H

#include <string>
#include <sstream>
#include <vector>

#define ENUM_MAP_DEFINE(type) std::string GetStringValue(const type &T);
#define ENUM_MAP(type, strings) std::string GetStringValue(const type &T) \
{ \
	return GetEnumName((int)T, strings); \
};

namespace NNTLib
{
	//bei hinzufügen eines neuen Enums in Enums.cpp den string anchziehen sonst wird dieses als UNKNOWN ausgegeben
	enum class FunctionEnum {LINEAR=0,LOGISTIC=1,TANH=2,LECUN_TANH=3};
	enum class MutateEnum {NONE=0,UNBIASED_MUTATA_WEIGHTS=1, BIAS_ED_MUTATE_WEIGHTS=2,MUTATE_NODES=3};
	enum class CrossoverEnum {NONE=0,CROSSOVER_WEIGHTS=1, CROSSOVER_NODES=2,CROSSOVER_ONEPOINT=3,CROSSOVER_TWOPOINT=4};
	enum class RouletteEnum {FITTNESSBASED=0,INDEXBASED=1};
	enum class WeightInitEnum {NONE=0,UNIFORM=1,LECUN=2,UNIFORM5=3,NORMAL5=4,DEBUGONE=5};//One zu test zwecken

	//definiere Funktionen GetStringValue um Enums als strings auszugeben
	ENUM_MAP_DEFINE(FunctionEnum);
	ENUM_MAP_DEFINE(MutateEnum);
	ENUM_MAP_DEFINE(CrossoverEnum);
	ENUM_MAP_DEFINE(RouletteEnum);
	ENUM_MAP_DEFINE(WeightInitEnum);

	std::string GetEnumName(unsigned int v, std::string tmp);
}
#endif
