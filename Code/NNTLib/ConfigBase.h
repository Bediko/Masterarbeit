#ifndef ConfigBase_H
#define ConfigBase_H

#include <iostream>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include "Enums.h"
class ConfigBase
{
private:
public:
	void LoadFile(const char* file);
	virtual ~ConfigBase();
	virtual void PrintData()=0;
	virtual bool IsConfigValid()=0;
protected:
	virtual void HandleNameValue(std::string name,std::string value)=0;
};

#endif
