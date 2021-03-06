#ifndef NeuralNetwork_H
#define NeuralNetwork_H
#include <random>

#include "Layer.h"
#include "DataContainer.h"

namespace NNTLib
{
	class NeuralNetwork
	{
	protected:
		void copy(const NeuralNetwork &that);
		std::mt19937 generator;
		void init();
		void freeMem();
	public:
		/// <summary>
		/// The layers count
		/// </summary>
		int LayersCount;
		/// <summary>
		/// The mean square error
		/// </summary>
		double MeanSquareError;
		/// <summary>
		/// The weight initialize type
		/// </summary>
		WeightInitEnum WeightInitType;
		/// <summary>
		/// The function type
		/// </summary>
		FunctionEnum FunctionType;
		FunctionEnum LastLayerFunction;
		/// <summary>
		/// The total neuron count
		/// </summary>
		int TotalNeuronCount;
		/// <summary>
		/// The layers
		/// </summary>
		Layer *Layers;
		NeuralNetwork();
		NeuralNetwork(int *layers,int layercount,WeightInitEnum initType,FunctionEnum functionType, FunctionEnum LastLayerFunction);


		//Rule of Three http://stackoverflow.com/questions/4172722/what-is-the-rule-of-three
		~NeuralNetwork();
		NeuralNetwork(const NeuralNetwork &that);
		NeuralNetwork& operator= (const NeuralNetwork &that);
		bool operator < (const NeuralNetwork& net) const;

		void InitWeights(WeightInitEnum initType);
		double GenerateRandomWeight(int weightCount);

		void SaveWeights(const std::string file);
		void LoadWeights(const std::string file);

		void Propagate(const double *input);

		void CalculateMSE(const DataContainer& data);
	};
}
#endif
