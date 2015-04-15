#ifndef TrainerBase_H
#define TrainerBase_H

//#include <stdexcept>
//#include "DataContainer.h"
#include "NeuralNetwork.h"
//#include "Functions.h"
#include "TrainingMeasure.h"

namespace NNTLib
{
	class TrainerBase
	{
	protected:
		void initMeasureResult(int lenght);
	public:
		/// <summary>
		/// The measure result
		/// </summary>
		TrainingMeasure * MeasureResult;
		/// <summary>
		/// The measure result lenght
		/// </summary>
		int MeasureResultLenght;
		/// <summary>
		/// The measure filled result lenght
		/// </summary>
		int MeasureFilledResultLenght;
		void SaveResult(std::string file,unsigned long long timeOffsetInMs=0);
		TrainerBase();
		~TrainerBase();
	};
}
#endif