#ifndef GeneticAlgorithm_H
#define GeneticAlgorithm_H
//#include <iostream>

#include "TrainerBase.h"

namespace NNTLib
{
	class GeneticAlgorithm : public TrainerBase
	{
	private:
		/// <summary>
		/// The population size
		/// </summary>
		int populationSize;
		/// <summary>
		/// The roulette sum
		/// </summary>
		double rouletteSum;
		void calculateRouletteSum(RouletteEnum mutateType,int skip=0);
		int rouletteIndexFittnesBased(int skip=0);
		int rouletteIndexRankBased(int skip=0);
		void sortSingleElement(int  index);
		/// <summary>
		/// The generator
		/// </summary>
		std::mt19937 generator;
		/// <summary>
		/// The next population
		/// </summary>
		NeuralNetwork **nextPopulation;

		void trainSteadyState(const DataContainer & dataContainer,int maxLoopCount,double errorThreshold,MutateEnum mutateType,CrossoverEnum crossoverType, RouletteEnum rouletteType,double mutationProbability,double crossoverProbability,int mutateNodeCount);
		void trainPopulation(const DataContainer & dataContainer,int maxLoopCount,double errorThreshold,MutateEnum mutateType,CrossoverEnum crossoverType, RouletteEnum rouletteType,int eltismCount,double mutationProbability,double crossoverProbability,int mutateNodeCount);

	public:
		/// <summary>
		/// The current population
		/// </summary>
		NeuralNetwork **currentPopulation;
		GeneticAlgorithm(NeuralNetwork **currentPopulation,int populationsize);
		~GeneticAlgorithm();

		//todo: parameter kapseln
		void Train(const DataContainer & dataContainer,int maxLoopCount,double errorThreshold,MutateEnum mutateType,CrossoverEnum crossoverType, RouletteEnum rouletteType,int eltismCount,double mutationProbability=0.1,double crossoverProbability=0.8,int mutateNodeCount=0);
	};
}
#endif
