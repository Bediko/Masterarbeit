#include "main.h"

#include "Neuron.h"

//#define EXAMPLE

int main(int argc, char* argv[]) {
	try {
#ifdef EXAMPLE
		SimpleXORExampleBackpropagation();
		//SimpleXORExampleGenetic();
		//SimpleMNISTExampleBackpropagation();
		//SimpleMNISTExampleGenetic();

		std::cin.get();
#endif

		BackpropagationConfig backpropConfig;
		NeuralNetworkConfig networkConfig;
		GeneticAlgorithmConfig geneticConfig;
		ContrastiveDivergenceConfig ContrastiveDivergenceConfig;
		NNTLib::DataContainer dataContainer;
		NNTLib::DataContainer additionalTestDataContainer;
		unsigned long long timeOffsetInMs = 0;
		int iterationCount = 1;

		int k = 1;
		int testCount = 0;
		int trainCount = 0;
		int partCount = 0;

		std::string backPropagationConfigFile = getCmdOptionValue(argv, argv + argc, "-B");
		std::string ContrastiveDivergenceConfigFile = getCmdOptionValue(argv, argv + argc, "-CD");
		std::string neuralNetworkConfigFile = getCmdOptionValue(argv, argv + argc, "-N");
		std::string geneticAlgorithmConfigFile = getCmdOptionValue(argv, argv + argc, "-G");
		std::string dataContainerFile = getCmdOptionValue(argv, argv + argc, "-D");
		std::string saveWeightsFile = getCmdOptionValue(argv, argv + argc, "-SW");
		std::string saveMSEFile = getCmdOptionValue(argv, argv + argc, "-SR");
		std::string loadWeightsFile = getCmdOptionValue(argv, argv + argc, "-LW");
		std::string additionaltestDataContainerFile = getCmdOptionValue(argv, argv + argc, "-TD");
		std::string errorDiffValue = getCmdOptionValue(argv, argv + argc, "-C");
		std::string kcrossValidation = getCmdOptionValue(argv, argv + argc, "-K");
		std::string offsetValue = getCmdOptionValue(argv, argv + argc, "-O");
		std::string iterationValue = getCmdOptionValue(argv, argv + argc, "-I");

		if (!offsetValue.empty()) {
			timeOffsetInMs = std::stoull(offsetValue, nullptr, 0);
		}

		if (!iterationValue.empty()) {
			iterationCount = atoi(iterationValue.c_str());
		}

		double *sumMSEBackprop = new double[iterationCount]();
		double *sumMSEGenetic = new double[iterationCount]();
		double *sumErrorTrain = new double[iterationCount]();
		double *sumErrorTest = new double[iterationCount]();
		double *sumErrorAdditional = new double[iterationCount]();

		//ohne neuronales netz kann man nichts machen also prüfen
		if (neuralNetworkConfigFile.empty()) {
			std::cout << ARGUMENT_MESSAGE << std::endl;
			return 0;
		}

		//Lade alle Konfig Dateien
		if (!neuralNetworkConfigFile.empty()) {
			std::cout << "Loading network config: " <<  neuralNetworkConfigFile << std::endl;
			networkConfig.LoadFile(neuralNetworkConfigFile.c_str());//neuralNetworkConfigFiles
			networkConfig.PrintData();
		}

		if (!backPropagationConfigFile.empty()) {
			std::cout << "Loading backpropagation config: " <<  backPropagationConfigFile << std::endl;
			backpropConfig.LoadFile(backPropagationConfigFile.c_str());
			backpropConfig.PrintData();
		}

		if (!geneticAlgorithmConfigFile.empty()) {
			std::cout << "Loading genetic config: " <<  geneticAlgorithmConfigFile << std::endl;
			geneticConfig.LoadFile(geneticAlgorithmConfigFile.c_str());
			geneticConfig.PrintData();
		}
		if (! ContrastiveDivergenceConfigFile.empty()) {
			std::cout << "Loading Contrastive Divergence config: " <<  ContrastiveDivergenceConfigFile << std::endl;
			ContrastiveDivergenceConfig.LoadFile(ContrastiveDivergenceConfigFile.c_str());
			ContrastiveDivergenceConfig.PrintData();
		}

		if (!additionaltestDataContainerFile.empty()) {
			std::cout << "Loading additional Test Data: " <<  additionaltestDataContainerFile << std::endl;
			additionalTestDataContainer.LoadFile(additionaltestDataContainerFile.c_str());
			std::cout << "additional testcount" << additionalTestDataContainer.DataCount << std::endl;
		}

		if (!dataContainerFile.empty()) {
			std::cout << "Loading Data: " <<  dataContainerFile << std::endl;
			dataContainer.LoadFile(dataContainerFile.c_str());

			if (!kcrossValidation.empty())
				k = atoi(kcrossValidation.c_str());

			partCount = dataContainer.DataCount / k;
			//falls nicht gerade ausgeht nehmen wir den rest einfach mit in die test daten

			if (k > 1)
				testCount = dataContainer.DataCount - (partCount * (k - 1));

			trainCount = dataContainer.DataCount - testCount;

			std::cout << "testcount" << testCount << std::endl;
			std::cout << "trainCount" << trainCount << std::endl;
		}

		for (int iteration = 0; iteration < iterationCount; ++iteration) {
			std::cout << LINE;

			for (int i = 0; i < k; ++i) {
				std::cout << LINE;

				NNTLib::NeuralNetwork result(networkConfig.LayerNeuronCount, networkConfig.LayerCount, networkConfig.WeightInitType, networkConfig.FunctionType);

				if (!loadWeightsFile.empty()) {
					result.LoadWeights(loadWeightsFile.c_str());
				}

				if (!dataContainerFile.empty()) {
					//Splitte Daten in Train und Test Daten für K Validierung
					NNTLib::DataContainer trainData;
					NNTLib::DataContainer testData;

					testData.Init(testCount, dataContainer.InputCount, dataContainer.OutputCount);
					testData.CopyData(dataContainer, 0, i * partCount, testCount);
					//std::cout << "test from/to: " <<i*partCount<< " / " <<(i*partCount+testCount) <<std::endl;

					trainData.Init(trainCount, dataContainer.InputCount, dataContainer.OutputCount);
					//bei 0 anfangen und bis zum test anfang kopieren
					trainData.CopyData(dataContainer, 0, 0, i * partCount);

					//dort anfangen wo test aufgehört hat und bis zum ende kopieren
					trainData.CopyData(dataContainer, i * partCount, (i * partCount) + testCount, dataContainer.DataCount - ((i * partCount) + testCount));

					if (geneticConfig.MaxLoopCount > 0 && geneticConfig.PopulationSize > 0) {
						std::cout << "start genetic algorithm: "  << std::endl;
						NNTLib::NeuralNetwork** population = new NNTLib::NeuralNetwork*[geneticConfig.PopulationSize];

						for (int p = 0; p < geneticConfig.PopulationSize; ++p) {
							population[p] = new NNTLib::NeuralNetwork(networkConfig.LayerNeuronCount, networkConfig.LayerCount, networkConfig.WeightInitType, networkConfig.FunctionType);
						}

						*population[0] = result;
						NNTLib::GeneticAlgorithm geneticAlg(population, geneticConfig.PopulationSize);
						geneticAlg.Train(trainData, geneticConfig.MaxLoopCount, geneticConfig.ErrorThreshold, geneticConfig.MutateType, geneticConfig.CrossType, geneticConfig.RouletteType, geneticConfig.EltismCount, geneticConfig.MutationProbability, geneticConfig.CrossoverProbability, geneticConfig.MutateNodeCount);
						result = *population[0]; //alle mit gleichen werten initialisieren macht kein sinn also nur den ersten

						for (int p = 0; p < geneticConfig.PopulationSize; ++p) {
							delete population[p];
						}
						delete [] population;

						if (geneticAlg.MeasureFilledResultLenght != 0) {
							//std::cout<< "index"<<geneticAlg.MeasureFilledResultLenght <<std::endl;;
							sumMSEGenetic[iteration] += geneticAlg.MeasureResult[geneticAlg.MeasureFilledResultLenght - 1].MeanSquareError;
							std::cout << "execute time in ms: " << geneticAlg.MeasureResult[geneticAlg.MeasureFilledResultLenght - 1].ExecuteTime << "\n";
							std::cout << "mse value: " << geneticAlg.MeasureResult[geneticAlg.MeasureFilledResultLenght - 1].MeanSquareError << "\n";
						}

						if (!saveMSEFile.empty()) {
							std::string filename(saveMSEFile);

							replace(filename, "%K", i);
							replace(filename, "%N", splitFileName(neuralNetworkConfigFile));
							replace(filename, "%B", "");
							replace(filename, "%G", splitFileName(geneticAlgorithmConfigFile));
							replace(filename, "%D", splitFileName(dataContainerFile));

							geneticAlg.SaveResult(filename.c_str(), timeOffsetInMs);
						}
					}
					if (ContrastiveDivergenceConfig.Epochs > 0) {
						double maxErrordiff = atof(errorDiffValue.c_str());
						NNTLib::DeepBeliefNet dbn(networkConfig.LayerNeuronCount, networkConfig.LayerCount, networkConfig.WeightInitType, networkConfig.FunctionType);
						NNTLib::ContrastiveDivergence CD(dbn);
						CD.Train(trainData, ContrastiveDivergenceConfig.LearnRate, ContrastiveDivergenceConfig.Epochs, ContrastiveDivergenceConfig.BatchSize, ContrastiveDivergenceConfig.GibbsSteps);

						dbn.SaveWeightsforNN("test" + std::to_string(iteration));
						result.LoadWeights("test" + std::to_string(iteration));						
//NNTLib::Backpropagation backpropAlg(result);
						  //std::cout << "SIGMOID BACKPROPAGATION" << std::endl;
						  //backpropAlg.Train(trainData, backpropConfig.Alpha, backpropConfig.MaxLoopCount, backpropConfig.Momentum, backpropConfig.BatchSize, backpropConfig.ErrorThreshold, backpropConfig.DecayRate);
						  //std::cout << "Test Traindata " << i + 1 << "/" << k << std::endl;
						  //int error = TestNetwork(maxErrordiff, &trainData, dbn);
						
						 // if (k > 1) {
						 // 	std::cout << "Test Testdata " << i + 1 << "/" << k << std::endl;
						 // 	error = TestNetwork(maxErrordiff, &testData, dbn);
						 // } else {
						 // 	std::cout << "Test additional Testdata " << i + 1 << "/" << k << std::endl;
						 // 	error = TestNetwork(maxErrordiff, &additionalTestDataContainer, dbn);
						 // }
						//result.LoadWeights("test" + std::to_string(iteration));


						// std::cout << "OHNE BACK PROPAGATION SIGMOID" << std::endl;
						// result.FunctionType = static_cast<NNTLib::FunctionEnum>(1);
						// std::cout << "Test Traindata " << i + 1 << "/" << k << std::endl;
						// error = TestNetwork(maxErrordiff, &trainData, result);
						// std::cout << "Test Testdata " << i + 1 << "/" << k << std::endl;
						// if (k > 1) {
						// 	error = TestNetwork(maxErrordiff, &testData, result);
						// } else {
						// 	error = TestNetwork(maxErrordiff, &additionalTestDataContainer, result);
						// }
						//  std::ofstream myfile;
						//  myfile.open("diff" + std::to_string(iteration));


						// for (int i = 0; i < result.LayersCount; ++i) {
						// 	NNTLib::Layer* layer = &result.Layers[i];

						// 	for (int j = 0; j < layer->NeuronCount; ++j) {
						// 		NNTLib::Neuron* neuron = &layer->Neurons[j];

						// 		for (int k = 0; k < layer->InputValuesCountWithBias; ++k) {
						// 			myfile << neuron->Weights[k] << " ";
						// 		}
						// 		myfile << "\n";
						// 	}
						// 	myfile << "\n";
						// }
						// myfile << error << "\n\n";


						// std::cout << "OHNE BACK PROPAGATION BINÄR" << std::endl;
						// result.FunctionType = static_cast<NNTLib::FunctionEnum>(4);
						// std::cout << "Test Traindata " << i + 1 << "/" << k << std::endl;
						// error = TestNetwork(maxErrordiff, &trainData, result);
						// std::cout << "Test Testdata " << i + 1 << "/" << k << std::endl;
						// if (k > 1) {
						// 	error = TestNetwork(maxErrordiff, &testData, result);
						// } else {
						// 	error = TestNetwork(maxErrordiff, &additionalTestDataContainer, result);
						// }
						// for (int i = 0; i < result.LayersCount; ++i) {
						// 	NNTLib::Layer* layer = &result.Layers[i];

						// 	for (int j = 0; j < layer->NeuronCount; ++j) {
						// 		NNTLib::Neuron* neuron = &layer->Neurons[j];

						// 		for (int k = 0; k < layer->InputValuesCountWithBias; ++k) {
						// 			myfile << neuron->Weights[k] << " ";
						// 		}
						// 		myfile << "\n";
						// 	}
						// 	myfile << "\n";
						// }
						// myfile << error << "\n\n";

						//result.FunctionType = static_cast<NNTLib::FunctionEnum>(1);
						//backpropAlg.Train(trainData, backpropConfig.Alpha, backpropConfig.MaxLoopCount, backpropConfig.Momentum, backpropConfig.BatchSize, backpropConfig.ErrorThreshold, backpropConfig.DecayRate);
						//std::cout << "MIT BACK PROPAGATION SIGMOID" << std::endl;
						//result.SaveWeights("BPsig" + std::to_string(iteration));
						//std::cout << "Test Traindata " << i + 1 << "/" << k << std::endl;
						//int error = TestNetwork(maxErrordiff, &trainData, result);
						//std::cout << "Test Testdata " << i + 1 << "/" << k << std::endl;
						//if (k > 1) {
						//	error = TestNetwork(maxErrordiff, &testData,result);
						//} else {
						//	error = TestNetwork(maxErrordiff, &additionalTestDataContainer,result);
						//}
						//for (int i = 0; i < result.LayersCount; ++i) {
						//	NNTLib::Layer* layer = &result.Layers[i];

						//	for (int j = 0; j < layer->NeuronCount; ++j) {
						//		NNTLib::Neuron* neuron = &layer->Neurons[j];

						//		for (int k = 0; k < layer->InputValuesCountWithBias; ++k) {
						//			myfile << neuron->Weights[k] << " ";
						//		}
						//		myfile << "\n";
						//	}
						//	myfile << "\n";
						//}
						//myfile << error << "\n\n";
						//result.LoadWeights("test" + std::to_string(iteration));
						//result.FunctionType = static_cast<NNTLib::FunctionEnum>(4);
						//backpropAlg.Train(trainData, backpropConfig.Alpha, backpropConfig.MaxLoopCount, backpropConfig.Momentum, backpropConfig.BatchSize, backpropConfig.ErrorThreshold, backpropConfig.DecayRate);
						//result.SaveWeights("BPbin" + std::to_string(iteration));
						// std::cout << "MIT BACK PROPAGATION BINÄR" << std::endl;
						// std::cout << "Test Traindata " << i + 1 << "/" << k << std::endl;
						// error = TestNetwork(maxErrordiff, &trainData, result);
						// if (k > 1) {
						// 	error = TestNetwork(maxErrordiff, &testData, result);
						// } else {
						// 	error = TestNetwork(maxErrordiff, &additionalTestDataContainer, result);
						// }
						// for (int i = 0; i < result.LayersCount; ++i) {
						// 	NNTLib::Layer* layer = &result.Layers[i];

						// 	for (int j = 0; j < layer->NeuronCount; ++j) {
						// 		NNTLib::Neuron* neuron = &layer->Neurons[j];

						// 		for (int k = 0; k < layer->InputValuesCountWithBias; ++k) {
						// 			myfile << neuron->Weights[k] << " ";
						// 		}
						// 		myfile << "\n";
						// 	}
						// 	myfile << "\n";
						// }
						// myfile << error << "\n\n";

						// myfile.close();
						//backpropConfig.MaxLoopCount=0;
					}

					if (backpropConfig.MaxLoopCount > 0) {
						std::cout << "start backpropagation algorithm: "  << std::endl;
						NNTLib::Backpropagation backpropAlg(result);

						backpropAlg.Train(trainData, backpropConfig.Alpha, backpropConfig.MaxLoopCount, backpropConfig.Momentum, backpropConfig.BatchSize, backpropConfig.ErrorThreshold, backpropConfig.DecayRate);

						if (backpropAlg.MeasureFilledResultLenght != 0) {
							//std::cout <<"c"<<backpropAlg.MeasureFilledResultLenght<<std::endl;
							sumMSEBackprop[iteration] += backpropAlg.MeasureResult[backpropAlg.MeasureFilledResultLenght - 1].MeanSquareError;
							std::cout << "execute time in ms: " << backpropAlg.MeasureResult[backpropAlg.MeasureFilledResultLenght - 1].ExecuteTime << "\n";
							std::cout << "mse value: " << backpropAlg.MeasureResult[backpropAlg.MeasureFilledResultLenght - 1].MeanSquareError << "\n";
						}

						if (!saveMSEFile.empty()) {
							std::string filename(saveMSEFile);

							replace(filename, "%K", i);
							replace(filename, "%N", splitFileName(neuralNetworkConfigFile));
							replace(filename, "%B", splitFileName(backPropagationConfigFile));
							replace(filename, "%G", "");
							replace(filename, "%D", splitFileName(dataContainerFile));

							backpropAlg.SaveResult(filename.c_str(), timeOffsetInMs);
						}
					}




					if (!saveWeightsFile.empty()) {
						std::string filename(saveWeightsFile);
						replace(filename, "%K", i);
						replace(filename, "%N", splitFileName(neuralNetworkConfigFile));
						replace(filename, "%B", splitFileName(backPropagationConfigFile));
						replace(filename, "%G", splitFileName(geneticAlgorithmConfigFile));
						replace(filename, "%D", splitFileName(dataContainerFile));

						result.SaveWeights(filename.c_str());
					}

					if (!errorDiffValue.empty()) {
						double maxErrordiff = atof(errorDiffValue.c_str());
						std::cout << "Test Traindata " << i + 1 << "/" << k << std::endl;
						sumErrorTrain[iteration] += TestNetwork(maxErrordiff, &trainData, result);

						if (k > 1) {
							std::cout << "Test Testdata " << i + 1 << "/" << k << std::endl;
							sumErrorTest[iteration] += TestNetwork(maxErrordiff, &testData, result);
						}
					}

					if (i == k - 1 && k > 1) {
						std::cout << LINE << std::endl;
						sumMSEBackprop[iteration] /= k;
						std::cout << "(sum mse backprop)/k :" << sumMSEBackprop[iteration] << std::endl;

						sumMSEGenetic[iteration] /= k;
						std::cout << "(sum mse genetic)/k :" << sumMSEGenetic[iteration] << std::endl;

						sumErrorTrain[iteration] /= k;
						std::cout << "(sum error Train)/k :" << sumErrorTrain[iteration] << std::endl;
						double errorProzentualTrain = (sumErrorTrain[iteration] / (double)trainData.DataCount);
						std::cout << "(sum error relativ Train)k : " << errorProzentualTrain << std::endl;

						sumErrorTest[iteration] /= k;
						std::cout << "(sum error Test)/k :" << sumErrorTest[iteration] << std::endl;
						double errorProzentualTest = (sumErrorTest[iteration] / (double)testData.DataCount);
						std::cout << "(sum error relativ Test)k : " << errorProzentualTest << std::endl;
					}
				}

				if (!errorDiffValue.empty()) {
					double maxErrordiff = atof(errorDiffValue.c_str());
					if (!additionaltestDataContainerFile.empty()) {
						std::cout << "Test Additional Testdata" << std::endl;
						sumErrorAdditional[iteration] += TestNetwork(maxErrordiff, &additionalTestDataContainer, result);
					}
				}

				if (i == k - 1 && k > 1) {
					if (!additionaltestDataContainerFile.empty()) {
						sumErrorAdditional[iteration] /= k;
						std::cout << "(sum error additional)/k :" << sumErrorAdditional[iteration] << std::endl;
						std::cout << "(sum error additional relativ)/k :" << (sumErrorAdditional[iteration] / (double)additionalTestDataContainer.DataCount) << std::endl;
					}
				}
			}
		}
		std::cout << LINE << LINE;

		if (iterationCount > 1) {
			if (backpropConfig.MaxLoopCount > 0) {
				std::cout << LINE;
				double mean = calculateMean(sumMSEBackprop, iterationCount);
				double meanVariant = calculateMeanVariant(mean, sumMSEBackprop, iterationCount);
				std::cout << "Mittelwert MSE backrpop:" << mean << std::endl;
				std::cout << "Streuung MSE backrpop:" <<  meanVariant << std::endl;
			}

			if (geneticConfig.MaxLoopCount > 0 && geneticConfig.PopulationSize > 0) {
				std::cout << LINE;
				double mean = calculateMean(sumMSEGenetic, iterationCount);
				double meanVariant = calculateMeanVariant(mean, sumMSEGenetic, iterationCount);
				std::cout << "Mittelwert MSE genetic:" << mean << std::endl;
				std::cout << "Streuung MSE genetic:" <<  meanVariant << std::endl;
			}

			if (trainCount != 0) {
				std::cout << LINE;
				double mean = calculateMean(sumErrorTrain, iterationCount);
				double meanVariant = calculateMeanVariant(mean, sumErrorTrain, iterationCount);
				std::cout << "Mittelwert Fehler Train:" << mean << std::endl;
				std::cout << "Mittelwert Relativer Fehler Train:" << (mean / (double)trainCount) << std::endl;
				std::cout << "Streuung Fehler Train:" <<  meanVariant << std::endl;
				std::cout << "Streuung Reletiver Fehler Train:" <<  (meanVariant / (double)trainCount) << std::endl;
			}

			if (testCount != 0) {
				std::cout << LINE;
				double mean = calculateMean(sumErrorTest, iterationCount);
				double meanVariant = calculateMeanVariant(mean, sumErrorTest, iterationCount);
				std::cout << "Mittelwert Fehler Test:" << mean << std::endl;
				std::cout << "Mittelwert Relativer Fehler Test:" << (mean / (double)testCount) << std::endl;
				std::cout << "Streuung Fehler Test:" <<  meanVariant << std::endl;
				std::cout << "Streuung Relativer Fehler Test:" <<  (meanVariant / (double)testCount) << std::endl;
			}

			if (!additionaltestDataContainerFile.empty()) {
				std::cout << LINE;
				double mean = calculateMean(sumErrorAdditional, iterationCount);
				double meanVariant = calculateMeanVariant(mean, sumErrorAdditional, iterationCount);
				std::cout << "Mittelwert Fehler Test (additional):" << mean << std::endl;
				std::cout << "Mittelwert Relativer Fehler Test (additional):" << (mean / (double)additionalTestDataContainer.DataCount) << std::endl;
				std::cout << "Streuung Fehler Test (additional):" <<  meanVariant << std::endl;
				std::cout << "Streuung Relativer Fehler Test (additional):" <<  (meanVariant / (double)additionalTestDataContainer.DataCount) << std::endl;
			}
		}

		std::cout << LINE << LINE << LINE;

		delete [] sumMSEBackprop;
		delete [] sumMSEGenetic;
		delete [] sumErrorTrain;
		delete [] sumErrorTest;
		delete [] sumErrorAdditional;
	} catch ( const std::exception & ex ) {
		std::cout << ex.what() << std::endl;
	}

	//sicherstellen das alles ausgegeben wird
	std::cout.flush();

#ifdef _DEBUG //windows spezifischer debugmode flag
	std::cin.get();
#endif

	return 0;
}

//Defines nur für die folgenden Beispiele relevant
#define POPULATIONSIZE 50
#define ELITISM 2

/// <summary>
/// Simples the xor example backpropagation.
/// </summary>
void SimpleXORExampleBackpropagation() {
	std::cout << "SimpleXORExample" << std::endl;
	NNTLib::DataContainer dataContainer;
	int maxLoopCount = 2500;

	//XOR Werte definieren und in container laden
	//x1|x2|y
	//-------
	//0 |0 |0
	//0 |1 |1
	//1 |0 |1
	//1 |1 |0
	double data[4][2 + 1] = {{0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0}};
	//double data[4][2+1] = {{-1,-1,-1}, {-1,1,1}, {1,-1,1},{1,1,-1}};

	//Datacontainer füllen / Alternativ einfach aus Datei laden dataContainer.LoadFile("Data/XOR//xor_0_to_1.data");
	dataContainer.Init(4, 2, 1);
	for (int i = 0; i < dataContainer.DataCount; ++i) {
		for (int j = 0; j < dataContainer.InputCount; j++)
			dataContainer.DataInput[i][j] = data[i][j];
		for (int j = 0; j < dataContainer.OutputCount; j++)
			dataContainer.DataOutput[i][j] = data[i][j + 2]; //+dataContainer.InputCount
	}

	std::cout << "DataContainer Finished!" << std::endl;

	//Definiere Netzwerk 2-3-1
	int layercount = 3;
	int *layerNeuronCount = new int[layercount];
	layerNeuronCount[0] = dataContainer.InputCount; //2
	layerNeuronCount[1] = 3;
	layerNeuronCount[2] = dataContainer.OutputCount; //1

	//netzwerk erstellen
	NNTLib::NeuralNetwork net(layerNeuronCount, layercount, NNTLib::WeightInitEnum::LECUN, NNTLib::FunctionEnum::LOGISTIC);

	std::cout << "start train" << std::endl;
	NNTLib::Backpropagation prop(net);

	//Start the neural network training
	prop.Train(dataContainer, 0.5, maxLoopCount, 0.9, 1);

	std::cout << "mse : " << prop.MeasureResult[prop.MeasureFilledResultLenght - 1].MeanSquareError << "\n";
	std::cout << "training took: " << prop.MeasureResult[prop.MeasureFilledResultLenght - 1].ExecuteTime << " ms to execute \n";

	//Vergleiche Ist und Soll
	for (int i = 0; i < dataContainer.DataCount; ++i) {
		net.Propagate(dataContainer.DataInput[i]);
		std::cout << "TESTED PATTERN " << i << " DESIRED OUTPUT: " << *dataContainer.DataOutput[i] << " NET RESULT: " << net.Layers[net.LayersCount - 1].Neurons[0].Output << std::endl;
	}
	//Gewichte und Ergebnisse pro Iteration speichern
	prop.SaveResult("Result.dat");
	net.SaveWeights("Weights.txt");

	delete [] layerNeuronCount;
	std::cout << "finished" << std::endl;
}

/// <summary>
/// Simples the xor example genetic.
/// </summary>
void SimpleXORExampleGenetic() {
	std::cout << "SimpleXORExampleGenetic" << std::endl;
	NNTLib::DataContainer dataContainer;
	int maxLoopCount = 100;

	//XOR Werte definieren und in container laden
	//x1|x2|y
	//-------
	//0 |0 |0
	//0 |1 |1
	//1 |0 |1
	//1 |1 |0
	double data[4][2 + 1] = {{0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0}};
	//double data[4][2+1] = {{-1,-1,-1}, {-1,1,1}, {1,-1,1},{1,1,-1}};

	//Datacontainer füllen / Alternativ einfach aus Datei laden dataContainer.LoadFile("Data/XOR//xor_0_to_1.data");
	dataContainer.Init(4, 2, 1);
	for (int i = 0; i < dataContainer.DataCount; ++i) {
		for (int j = 0; j < dataContainer.InputCount; j++)
			dataContainer.DataInput[i][j] = data[i][j];
		for (int j = 0; j < dataContainer.OutputCount; j++)
			dataContainer.DataOutput[i][j] = data[i][j + 2]; //+dataContainer.InputCount
	}

	std::cout << "DataContainer Finished!" << std::endl;

	//Definiere Netzwerk 2-3-1
	int layercount = 3;
	int *layerNeuronCount = new int[layercount];
	layerNeuronCount[0] = dataContainer.InputCount; //2
	layerNeuronCount[1] = 3;
	layerNeuronCount[2] = dataContainer.OutputCount; //1

	NNTLib::NeuralNetwork** population = new NNTLib::NeuralNetwork*[POPULATIONSIZE];

	for (int p = 0; p < POPULATIONSIZE; ++p) {
		population[p] = new NNTLib::NeuralNetwork(layerNeuronCount, layercount, NNTLib::WeightInitEnum::UNIFORM, NNTLib::FunctionEnum::LOGISTIC);
	}

	std::cout << "start train" << std::endl;
	NNTLib::GeneticAlgorithm genetic(population, POPULATIONSIZE);

	//Start the neural network training
	genetic.Train(dataContainer, maxLoopCount, 0, NNTLib::MutateEnum::MUTATE_NODES, NNTLib::CrossoverEnum::CROSSOVER_NODES, NNTLib::RouletteEnum::FITTNESSBASED, ELITISM, 0.05, 0.8, 2);

	std::cout << "mse : " << genetic.MeasureResult[genetic.MeasureFilledResultLenght - 1].MeanSquareError << "\n";
	std::cout << "training took: " << genetic.MeasureResult[genetic.MeasureFilledResultLenght - 1].ExecuteTime << " ms to execute \n";

	//Vergleiche Ist und Soll
	for (int i = 0; i < dataContainer.DataCount; ++i) {
		population[0]->Propagate(dataContainer.DataInput[i]);
		std::cout << "TESTED PATTERN " << i << " DESIRED OUTPUT: " << *dataContainer.DataOutput[i] << " NET RESULT: " << population[0]->Layers[population[0]->LayersCount - 1].Neurons[0].Output << std::endl;
	}

	delete [] layerNeuronCount;
	std::cout << "finished" << std::endl;
}

/// <summary>
/// Simples the mnist example backpropagation.
/// </summary>
void  SimpleMNISTExampleBackpropagation() {
	int maxLoopCount = 100;
	NNTLib::DataContainer container;
	container.LoadFile("Data/MNIST_TRAIN_01_09");

	NNTLib::DataContainer testcontainer;
	testcontainer.LoadFile("Data/MNIST_TEST_01_09");

	//DisplaySingleMNISTChar(&container,0);

	std::cout << "DataContainer Finished!" << std::endl;
	int layercount = 3;

	int *layerNeuronCount = new int[layercount];

	layerNeuronCount[0] = container.InputCount;
	layerNeuronCount[1] = 300;
	layerNeuronCount[2] = container.OutputCount;
	//int lastError =testcontainer.DataCount;

	NNTLib::NeuralNetwork net(layerNeuronCount, layercount, NNTLib::WeightInitEnum::UNIFORM, NNTLib::FunctionEnum::LOGISTIC);

	//net.LoadWeights("Result/W_L");

	NNTLib::Backpropagation prop(net);

	//We create the network
	std::cout << "start train" << std::endl;
	//Start the neural network training
	prop.Train(container, 0.5f, maxLoopCount);
	std::cout << "training took: " << prop.MeasureResult[prop.MeasureFilledResultLenght - 1].ExecuteTime << " ms to execute \n";
	std::cout << "mse value: " << prop.MeasureResult[prop.MeasureFilledResultLenght - 1].MeanSquareError << "\n";

	TestNetwork(0.5, &testcontainer, net);

	//testen wieviele falsch erkannt werden

	delete [] layerNeuronCount;
}

/// <summary>
/// Simples the mnist example genetic.
/// </summary>
void  SimpleMNISTExampleGenetic() {
	int maxLoopCount = 10;
	NNTLib::DataContainer container;
	container.LoadFile("Data/MNIST_TRAIN_0_1");

	NNTLib::DataContainer testcontainer;
	testcontainer.LoadFile("Data/MNIST_TEST_0_1");

	std::cout << "DataContainer Finished!" << std::endl;
	int layercount = 3;

	int *layerNeuronCount = new int[layercount];

	layerNeuronCount[0] = container.InputCount;
	layerNeuronCount[1] = 300;
	layerNeuronCount[2] = container.OutputCount;
	//int lastError =testcontainer.DataCount;

	NNTLib::NeuralNetwork** population = new NNTLib::NeuralNetwork*[POPULATIONSIZE];

	for (int p = 0; p < POPULATIONSIZE; ++p) {
		population[p] = new NNTLib::NeuralNetwork(layerNeuronCount, layercount, NNTLib::WeightInitEnum::UNIFORM, NNTLib::FunctionEnum::LOGISTIC);
	}

	NNTLib::GeneticAlgorithm genetic(population, POPULATIONSIZE);

	//Start the neural network training
	genetic.Train(container, maxLoopCount, 0, NNTLib::MutateEnum::MUTATE_NODES, NNTLib::CrossoverEnum::CROSSOVER_NODES, NNTLib::RouletteEnum::FITTNESSBASED, ELITISM, 0.1, 0.8, 2);

	std::cout << "mse : " << genetic.MeasureResult[genetic.MeasureFilledResultLenght - 1].MeanSquareError << "\n";
	std::cout << "training took: " << genetic.MeasureResult[genetic.MeasureFilledResultLenght - 1].ExecuteTime << " ms to execute \n";

	//testen wieviele falsch erkannt werden
	TestNetwork(0.5, &testcontainer, *population[0]);

	delete [] layerNeuronCount;
}

/// <summary>
/// Displays the single mnist character.
/// </summary>
/// <param name="dataContainermnistTest">The data containermnist test.</param>
/// <param name="index">The index.</param>
void DisplaySingleMNISTChar(const NNTLib::DataContainer *dataContainermnistTest, int index) {
	for (int i = 0; i < 28; i++) {
		for (int j = 0; j < 28; j++) {
			if (dataContainermnistTest->DataInput[index][i * 28 + j] > 0)
				std::cout << "1";
			else
				std::cout << "0";
		}
		std::cout << "\n" ;
	}

	for (int i = 0; i < 10; i++) {
		std::cout << dataContainermnistTest->DataOutput[index][i];
	}
}

//void TEST()
//{
//	BackpropagationConfig backpropConfig;
//	backpropConfig.LoadFile("Config/B3-2");
//	NeuralNetworkConfig networkConfig;
//	networkConfig.LoadFile("Config/AR_N2");
//	NNTLib::DataContainer trainData;
//	NNTLib::DataContainer testData;
//	trainData.LoadFile("Data/AR_01_to_09_Train2");
//	testData.LoadFile("Data/AR_01_to_09_Test2");
//	/*trainData2.LoadFile("Data/AR_0_to_1_Train");
//	testData2.LoadFile("Data/AR_0_to_1_Test");
//	trainData3.LoadFile("Data/AR_0_to_1_Train");
//	testData3.LoadFile("Data/AR_0_to_1_Test");
//	trainData4.LoadFile("Data/AR_0_to_1_Train");
//	testData4.LoadFile("Data/AR_0_to_1_Test");
//	trainData5.LoadFile("Data/AR_0_to_1_Train");
//	testData5.LoadFile("Data/AR_0_to_1_Test");*/
//
//	NNTLib::NeuralNetwork result(networkConfig.LayerNeuronCount,networkConfig.LayerCount,networkConfig.WeightInitType,networkConfig.FunctionType);
//	result.LoadWeights("Result/W_L");
//	NNTLib::Backpropagation backpropAlg(result);
////	int lasterrortest = testData.DataCount;
////	int lasterrortrain = trainData.DataCount;
////	double lastmse =1;
//	for(int i=0;i< 100;i++)
//	{
//		backpropAlg.Train(trainData,backpropConfig.Alpha,100,backpropConfig.Momentum,backpropConfig.BatchSize,backpropConfig.ErrorThreshold,backpropConfig.DecayRate);
//
//		double mse= backpropAlg.MeasureResult[backpropAlg.MeasureFilledResultLenght-1].MeanSquareError;
//		std::cout << "mse value: " <<mse  << "\n";
//
////		int errortrain = TestNetwork(0.5f,&trainData,result);
////		int errortest = TestNetwork(0.5f,&testData,result);
//
//		//if(errortest > lasterrortest && mse < lastmse && errortrain < lasterrortrain) //failed
//		//{
//		//	std::cout << "iteration count" << i <<std::endl;
//		//	break;
//		//}
//		//lastmse = mse;
//		//lasterrortrain = errortrain;
//		//lasterrortest = errortest;
//		//std::cout << "error: " <<errorcount<<std::endl;
//		//
//	}
//	std::cin.get();
//}
