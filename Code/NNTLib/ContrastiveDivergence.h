#ifndef ContrastiveDivergence_H
#define ContrastiveDivergence_H

#include "TrainerBase.h"
#include "DBNLayer.h"
namespace NNTLib {
class ContrastiveDivergence: public TrainerBase {
private:
	std::default_random_engine generator;
	void trainIncremental(const DataContainer &container, const double learnRate, const int Epochs, const int gibbs);
	int Binary(double x); //Matrix für Statistik
	double **statisticsdatav;
	double **statisticsdatah;
	double **statisticsmodelv;
	double **statisticsmodelh;
	double **statisticsdatasm;
	double **statisticsmodelsm;
	DBNLayer *bottom, *top;

//		void trainBatch(const DataContainer &container,const double learnRate,const int maxLoopCount,const double momentum,int minibatchSize,const double errorThreshold,const double decayRate);
public:
	/// <summary>
	/// The network
	/// </summary>
	DeepBeliefNet *network;
	unsigned long long runtime;
	void GibbsSampling(int gibbssteps, int softmax);
	void UpdateHiddenUnits(int softmax);
	void UpdateVisibleUnits(int softmax);

	ContrastiveDivergence(DeepBeliefNet &net);
	~ContrastiveDivergence();
	void Train(const DataContainer &container, const double learnRate, const int Epochs, int BatchSize = 1, int gibbs = 1);
};
}
#endif