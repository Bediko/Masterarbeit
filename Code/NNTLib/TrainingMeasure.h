#ifndef NEURALNETWORKMEASURE_H
#define NEURALNETWORKMEASURE_H

#ifdef WIN32
#include <Windows.h>
#else
#include <sys/time.h>
#include <ctime>
#endif

namespace NNTLib
{
	struct TrainingMeasure
	{
		/// <summary>
		/// The mean square error
		/// </summary>
		double MeanSquareError;
		/// <summary>
		/// The execute time
		/// </summary>
		unsigned long long ExecuteTime;
		TrainingMeasure();
		~TrainingMeasure();
	};

	unsigned long long GetTimeMs64();
}
#endif