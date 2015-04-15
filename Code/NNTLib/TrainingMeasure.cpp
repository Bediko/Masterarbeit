#include "TrainingMeasure.h"
namespace NNTLib
{
	/// <summary>
	/// Initializes a new instance of the <see cref="TrainingMeasure" /> struct.
	/// </summary>
	TrainingMeasure::TrainingMeasure()
	{
		MeanSquareError=0;
		ExecuteTime=0;
	}

	/// <summary>
	/// Finalizes an instance of the <see cref="TrainingMeasure" /> class.
	/// </summary>
	TrainingMeasure::~TrainingMeasure()
	{
	}

	/// <summary>
	/// Gets the time MS64.
	/// Source: http://stackoverflow.com/questions/1861294/how-to-calculate-execution-time-of-a-code-snippet-in-c
	/// mit modifikationen aus den kommentaren
	/// </summary>
	/// <returns></returns>
	unsigned long long GetTimeMs64()
	{
#ifdef WIN32
		/* Windows */
		FILETIME ft;
		LARGE_INTEGER li;

		/* Get the amount of 100 nano seconds intervals elapsed since January 1, 1601 (UTC) and copy it
		* to a LARGE_INTEGER structure. */
		GetSystemTimeAsFileTime(&ft);
		li.LowPart = ft.dwLowDateTime;
		li.HighPart = ft.dwHighDateTime;

		unsigned long long ret = li.QuadPart;
		ret -= 116444736000000000LL; /* Convert from file time to UNIX epoch time. */
		ret /= 10000; /* From 100 nano seconds (10^-7) to 1 millisecond (10^-3) intervals */

		return ret;
#else
		/* Linux */
		struct timeval tv;

		gettimeofday(&tv, nullptr);

		unsigned long long ret = tv.tv_usec;
		/* Convert from micro seconds (10^-6) to milliseconds (10^-3) */
		ret /= 1000;

		/* Adds the seconds (10^0) after converting them to milliseconds (10^-3) */
		ret += (tv.tv_sec * 1000);

		return ret;
#endif
	}
}