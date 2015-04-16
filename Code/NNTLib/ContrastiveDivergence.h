#ifndef ContrastiveDivergence_H
#define ContrastiveDivergence_H

#include "TrainerBase.h"
namespace NNTLib
{
	class ContrastiveDivergence: public TrainerBase
	{
	private:
		void trainIncremental(const DataContainer &container,const double learnRate,const int Epochs);
//		void trainBatch(const DataContainer &container,const double learnRate,const int maxLoopCount,const double momentum,int minibatchSize,const double errorThreshold,const double decayRate);
	public:
		/// <summary>
		/// The network
		/// </summary>
		NeuralNetwork *network;
		ContrastiveDivergence(NeuralNetwork &net);
		~ContrastiveDivergence();
		void Train(const DataContainer &container,const double learnRate,const int Epochs,int BatchSize=1);
	};
}
#endif