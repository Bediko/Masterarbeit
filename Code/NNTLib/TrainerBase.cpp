#include "TrainerBase.h"
namespace NNTLib
{
	/// <summary>
	/// Initializes the measure result.
	/// </summary>
	/// <param name="lenght">The lenght.</param>
	void TrainerBase::initMeasureResult(int lenght)
	{
		if(MeasureResult != nullptr)
		{
			delete [] MeasureResult;
			MeasureResult = nullptr;
		}
		MeasureResultLenght=lenght;
		MeasureResult = new TrainingMeasure[lenght]();
		MeasureFilledResultLenght =0;
	}

	/// <summary>
	/// Initializes a new instance of the <see cref="TrainerBase" /> class.
	/// </summary>
	TrainerBase::TrainerBase()
	{
		MeasureResult=nullptr;
	}

	/// <summary>
	/// Finalizes an instance of the <see cref="TrainerBase" /> class.
	/// </summary>
	TrainerBase::~TrainerBase()
	{
		delete [] MeasureResult;
	}

	/// <summary>
	/// Saves the mse result.
	/// </summary>
	/// <param name="file">The file.</param>
	/// <param name="timeOffsetInMs">The time offset in ms.</param>
	void TrainerBase::SaveResult(std::string file,unsigned long long timeOffsetInMs)
	{
		std::remove(file.c_str());

		std::ofstream myfile;
		myfile.open(file);

		if(!myfile)
		{
			std::string buf("Could not open file ");
			buf.append(file);
			throw std::runtime_error(buf);
		}

		for(int i=0;i<MeasureFilledResultLenght;++i)
		{
			myfile << i <<" "<< (MeasureResult[i].ExecuteTime + timeOffsetInMs) <<" "<< MeasureResult[i].MeanSquareError << "\n";
		}

		myfile.close();
	}
}