#include "ConfigBase.h"

/// <summary>
/// Finalizes an instance of the <see cref="ConfigBase" /> class.
/// </summary>
ConfigBase::~ConfigBase()
{}

/// <summary>
/// Loads the file.
/// </summary>
/// <param name="file">The file.</param>
void ConfigBase::LoadFile(const char* file)
{
	std::ifstream iFile;
	//iFile.exceptions ( ifstream::failbit | ifstream::badbit );//feuert aus mir unklaren gründen auch bei eof bei getline(..)?!

	iFile.open(file);

	if (!iFile)
	{
		std::string buf("Could not open file");
		buf.append(file);
		throw std::runtime_error(buf);
	}

	std::string line;

	while(getline(iFile, line))
	{
		if(line[0] == '#')//Zeile fängt mit # an, ist ein kommentar überspringen
			continue;

		std::stringstream streamInput(line);
		std::string name;
		std::string value;
		getline(streamInput, name, '=');
		getline(streamInput, value);

		HandleNameValue(name,value);
	}

	iFile.close();
}