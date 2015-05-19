#ifndef HELPER_H
#define HELPER_H

#include <string>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>

#include "NeuralNetwork.h"
#include "Backpropagation.h"
#include "GeneticAlgorithm.h"
#include "DataContainer.h"
#include "ContrastiveDivergence.h"
#include "DeepBeliefNet.h"

std::string getCmdOptionValue(char ** begin, char ** end, const std::string & option);

bool cmdOptionExists(char** begin, char** end, const std::string& option);

bool replace(std::string& str, const std::string& from, const int to);
bool replace(std::string& str, const std::string& from, const std::string& to);

std::string splitFileName(std::string str);

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems);

double calculateMean(double * values,int lenght);
double calculateMeanVariant(double mean, double * values,int lenght);

int TestNetwork(double maxErrordiff,const NNTLib::DataContainer * testDataContainer,NNTLib::NeuralNetwork  net);

#endif