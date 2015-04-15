#include "Helper.h"

/// <summary>
/// Gets the command value.
/// </summary>
/// <param name="begin">The begin.</param>
/// <param name="end">The end.</param>
/// <param name="option">The option.</param>
/// <returns></returns>
std::string getCmdOptionValue(char ** begin, char ** end, const std::string & option)
{
	char ** itr = std::find(begin, end, option);

	if (itr != end && ++itr != end)
	{
		return *itr;
	}
	return "";
}

/// <summary>
/// Commands the option exists.
/// </summary>
/// <param name="begin">The begin.</param>
/// <param name="end">The end.</param>
/// <param name="option">The option.</param>
/// <returns></returns>
bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
	return std::find(begin, end, option) != end;
}

/// <summary>
/// Splits the specified s.
/// </summary>
/// <param name="s">The s.</param>
/// <param name="delim">The delimiter.</param>
/// <param name="elems">The elems.</param>
/// <returns></returns>
std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems)
{
	std::stringstream ss(s+' ');
	std::string item;
	while(std::getline(ss, item, delim))
	{
		elems.push_back(item);
	}
	return elems;
}

/// <summary>
/// Splits the name of the file.
/// </summary>
/// <param name="str">The string.</param>
/// <returns></returns>
std::string splitFileName(std::string str)
{
	std::string copy(str);
	size_t end_pos = copy.find_last_of("/\\");

	if(end_pos != std::string::npos)
		copy.erase(copy.begin(),copy.begin()+end_pos+1);

	return copy;
}

/// <summary>
/// Replaces the specified string.
/// </summary>
/// <param name="str">The string.</param>
/// <param name="from">From.</param>
/// <param name="to">To.</param>
/// <returns></returns>
bool replace(std::string& str, const std::string& from, const std::string& to) {
	if(str.empty())
		return false;

	size_t start_pos = str.find(from);
	if(start_pos == std::string::npos)
		return false;
	str.replace(start_pos, from.length(), to);
	return true;
}

/// <summary>
/// Replaces the specified string.
/// </summary>
/// <param name="str">The string.</param>
/// <param name="from">From.</param>
/// <param name="to">To.</param>
/// <returns></returns>
bool replace(std::string& str, const std::string& from, const int to) {
	std::ostringstream oss;
	oss << to;
	return replace(str,from,oss.str());
}

/// <summary>
/// Calculates the mean.
/// </summary>
/// <param name="values">The values.</param>
/// <param name="lenght">The lenght.</param>
/// <returns></returns>
double calculateMean(double * values,int lenght)
{
	double mean=0;
	for(int i=0;i<lenght;i++)
	{
		mean+=values[i];
	}
	return (mean/lenght);
}

/// <summary>
/// Calculates the mean variant.
/// </summary>
/// <param name="mean">The mean.</param>
/// <param name="values">The values.</param>
/// <param name="lenght">The lenght.</param>
/// <returns></returns>
double calculateMeanVariant(double mean, double * values,int lenght)
{
	double meanVariant=0;
	for(int i=0;i<lenght;i++)
	{
		meanVariant+= ((values[i]-mean)*(values[i]-mean));//std::pow(values[i]-mean,2.);
	}
	return std::sqrt(meanVariant/lenght);
}

/// <summary>
/// Tests the network.
/// </summary>
/// <param name="maxErrordiff">The maximum errordiff.</param>
/// <param name="testDataContainer">The test data container.</param>
/// <param name="net">The net.</param>
/// <returns></returns>
int TestNetwork(double maxErrordiff,const NNTLib::DataContainer * testDataContainer,NNTLib::NeuralNetwork  net)
{
	int errorCounter =0;

	for(int i=0;i<testDataContainer->DataCount;++i)
	{
		net.Propagate(testDataContainer->DataInput[i]);

		if(testDataContainer->OutputCount == 1)
		{
			if(std::abs(net.Layers[net.LayersCount-1].Neurons[0].Output -  testDataContainer->DataOutput[i][0]) >= maxErrordiff )
			{
				errorCounter++;
				/*std::cout << "PATTERN :";
				for(int j=0;j<testDataContainer->InputCount;j++)
				{
				std::cout << testDataContainer->DataInput[i][j]<< " , ";
				}

				std::cout << " DESIRED OUTPUT: " <<testDataContainer->DataOutput[i][0]  << " NET RESULT: "<< 	net.Layers[net.LayersCount-1].Neurons[0].Output			<<std::endl;
				*/
			}
		}
		else
		{
			int indexMaxNeuron=0;
			//finde den höchsten Ausgabe index
			for(int j=0;j<testDataContainer->OutputCount;j++)
			{
				if(net.Layers[net.LayersCount-1].Neurons[indexMaxNeuron].Output < net.Layers[net.LayersCount-1].Neurons[j].Output)
				{
					indexMaxNeuron = j;
				}
			}

			int indexMaxData=0;

			//finde höchsten daten index (aufgrund der standard enkodierung ist das auch die Lösung)
			for(int j=0;j<testDataContainer->OutputCount;j++)
			{
				if(testDataContainer->DataOutput[i][indexMaxData] < testDataContainer->DataOutput[i][j])
				{
					indexMaxData = j;
				}
			}

			if(indexMaxNeuron != indexMaxData)
				errorCounter++;
		}
	}

	std::cout << "error count absolut: " << errorCounter <<std::endl;
	double errorProzentual = (errorCounter/(double)testDataContainer->DataCount);
	std::cout << "error count relativ: " << errorProzentual <<std::endl;
	return errorCounter;
}