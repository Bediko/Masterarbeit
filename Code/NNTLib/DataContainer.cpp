#include "DataContainer.h"
namespace NNTLib
{
	/// <summary>
	/// Initializes a new instance of the <see cref="DataContainer" /> struct.
	/// </summary>
	DataContainer::DataContainer()
	{
		init();
	}

	/// <summary>
	/// Initializes a new instance of the <see cref="DataContainer"/> class.
	/// </summary>
	/// <param name="that">The that.</param>
	DataContainer::DataContainer(const DataContainer &that)
	{
		init();
		copy(that);
	}
	/// <summary>
	/// Operator=s the specified that.
	/// </summary>
	/// <param name="that">The that.</param>
	/// <returns></returns>
	DataContainer& DataContainer::operator= (const DataContainer &that)
	{
		if (&that != this) {
			freeMem();
			init();
			copy(that);
		}
		return *this;
	}

	/// <summary>
	/// Finalizes an instance of the <see cref="DataContainer" /> class.
	/// </summary>
	DataContainer::~DataContainer()
	{
		freeMem();
	}

	/// <summary>
	/// Initializes this instance.
	/// </summary>
	void DataContainer::init()
	{
		DataCount=0;
		InputCount=0;
		OutputCount=0;
		DataInput=nullptr;
		DataOutput=nullptr;
	}

	/// <summary>
	/// Frees the memory.
	/// </summary>
	void DataContainer::freeMem()
	{
		if(DataInput)
		{
			for(int i=0;i<DataCount;i++)
			{
				delete [] DataInput[i];
			}
			delete [] DataInput;
		}

		if(DataOutput)
		{
			for(int i=0;i<DataCount;i++)
			{
				delete [] DataOutput[i];
			}
			delete [] DataOutput;
		}
	}

	/// <summary>
	/// Copies the specified that.
	/// </summary>
	/// <param name="that">The that.</param>
	void DataContainer::copy(const DataContainer &that)
	{
		Init(that.DataCount,that.InputCount,that.OutputCount);

		for(int i=0;i<that.DataCount;i++)
		{
			for(int m =0;m<InputCount;++m)
			{
				DataInput[i][m]=that.DataInput[i][m];
			}

			for(int n =0;n<OutputCount;++n)
			{
				DataOutput[i][n]=that.DataOutput[i][n];
			}
		}
	}

	/// <summary>
	/// Copies the data.
	/// </summary>
	/// <param name="src">The source.</param>
	/// <param name="startindexDst">The startindex DST.</param>
	/// <param name="startindexSource">The startindex source.</param>
	/// <param name="lenght">The lenght.</param>
	void DataContainer::CopyData(const DataContainer &src,int startindexDst,int startindexSource,int lenght)
	{
		if(startindexDst+lenght > src.DataCount || startindexDst < 0 || lenght < 0 || startindexSource < 0)
			throw std::runtime_error("datacontainer copy out of range");

		for(int i=0;i<lenght;i++)
		{
			for(int m =0;m<InputCount;++m)
			{
				DataInput[i+startindexDst][m]=src.DataInput[i+startindexSource][m];
			}

			for(int n =0;n<OutputCount;++n)
			{
				DataOutput[i+startindexDst][n]=src.DataOutput[i+startindexSource][n];
			}
		}
	}

	/// <summary>
	/// Initializes the specified data count.
	/// </summary>
	/// <param name="dataCount">The data count.</param>
	/// <param name="inputCount">The input count.</param>
	/// <param name="outputCount">The output count.</param>
	void DataContainer::Init(int dataCount,int inputCount,int outputCount)
	{
		freeMem();
		DataCount=dataCount;
		InputCount=inputCount;
		OutputCount=outputCount;

		if(DataCount > 0)
		{
			DataInput = new double*[DataCount];
			for(int i = 0; i < DataCount; ++i)
			{
				DataInput[i] = new double[InputCount]();
			}
		}

		if(OutputCount > 0)
		{
			DataOutput = new double*[DataCount];
			for(int i = 0; i < DataCount; ++i)
			{
				DataOutput[i] = new double[OutputCount]();
			}
		}
	}

	/// <summary>
	/// Loads the file.
	/// Datei muss wie folgt ausgebaut sein
	/// 2 2 1 //[Anzahl Datan] [Anzahl Input] [Anzahl Output]
	/// 0 0   //Input Daten 1 {0,0}
	/// 0	 //Output Daten 1 {0}
	/// 0 1	 //Input Daten 2 {0,1}
	/// 1	 //Output Daten 2 {1}
	/// </summary>
	/// <param name="file">The file.</param>
	void DataContainer::LoadFile(const char* file)
	{
		std::ifstream iFile;
		iFile.open(file);

		if (!iFile)
		{
			std::string buf("Could not open file");
			buf.append(file);
			throw std::runtime_error(buf);
		}

		std::string line;
		getline(iFile, line);
		std::stringstream stream(line);
		std::string dataCount;
		getline(stream, dataCount, ' ');
		std::string inputCount;
		getline(stream, inputCount, ' ');
		std::string outputCount;
		getline(stream, outputCount, ' ');

		int dataCountValue=atoi(dataCount.c_str());
		int inputCountValue=atoi(inputCount.c_str());
		int outputCountValue=atoi(outputCount.c_str());

		Init(dataCountValue,inputCountValue,outputCountValue);

		for(int l=0;l<DataCount;++l)
		{
			getline(iFile, line);
			std::stringstream streamInput(line);
			std::string singleInput;

			for(int m =0;m<InputCount;++m)
			{
				getline(streamInput, singleInput, ' ');
				DataInput[l][m]=atof(singleInput.c_str());
			}

			getline(iFile, line);
			std::stringstream streamOutput(line);

			for(int n =0;n<OutputCount;++n)
			{
				getline(streamOutput, singleInput, ' ');
				DataOutput[l][n]=atof(singleInput.c_str());
			}
		}

		iFile.close();
	}
}