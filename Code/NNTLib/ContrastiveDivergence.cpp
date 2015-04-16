#include "ContrastiveDivergence.h"
#include <iostream>

namespace NNTLib
{

	void ContrastiveDivergence::trainIncremental(const DataContainer &container,const double learnRate,const int Epochs){
		std::cout<<"train CD incremental"<<std::endl;
	}
	ContrastiveDivergence::ContrastiveDivergence(NeuralNetwork &net){
		this->network = &net;
	}
	ContrastiveDivergence::~ContrastiveDivergence(){};
	void ContrastiveDivergence::Train(const DataContainer &container,const double learnRate,const int Epochs,int BatchSize){
		std::cout<<"Train CD"<<std::endl;
		if(BatchSize==1){
				trainIncremental(container,learnRate,Epochs);
				return;
		}
	}
}
