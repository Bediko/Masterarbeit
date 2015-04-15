#ifndef Backpropagation_H
#define Backpropagation_H

#include "TrainerBase.h"
namespace NNTLib
{
	class Backpropagation: public TrainerBase
	{
	private:
		void trainIncremental(const DataContainer &container,const double learnRate,const int maxLoopCount,const double momentum,const double errorThreshold,const double decayRate);
		void trainBatch(const DataContainer &container,const double learnRate,const int maxLoopCount,const double momentum,int minibatchSize,const double errorThreshold,const double decayRate);
	public:
		/// <summary>
		/// The network
		/// </summary>
		NeuralNetwork *network;
		Backpropagation(NeuralNetwork &net);
		~Backpropagation();
		void Train(const DataContainer &container,const double learnRate,const int maxLoopCount,const double momentum=0,int minibatchSize=1,const double errorThreshold=0,const double decayRate=0);
	};
}
#endif