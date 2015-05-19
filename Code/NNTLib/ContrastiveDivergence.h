#ifndef ContrastiveDivergence_H
#define ContrastiveDivergence_H

#include "TrainerBase.h"
namespace NNTLib
{
	class ContrastiveDivergence: public TrainerBase
	{
	private:
		std::default_random_engine generator;
		void trainIncremental(const DataContainer &container,const double learnRate,const int Epochs);
		int Binary(double x);
//		void trainBatch(const DataContainer &container,const double learnRate,const int maxLoopCount,const double momentum,int minibatchSize,const double errorThreshold,const double decayRate);
	public:
		/// <summary>
		/// The network
		/// </summary>
		DeepBeliefNet *network;
		ContrastiveDivergence(DeepBeliefNet &net);
		~ContrastiveDivergence();
		void Train(const DataContainer &container,const double learnRate,const int Epochs,int BatchSize=1);
	};
}
#endif