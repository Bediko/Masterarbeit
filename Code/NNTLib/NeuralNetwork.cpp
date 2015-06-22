#include "NeuralNetwork.h"
#include <iostream>

namespace NNTLib {


//<summary>
//Empty Constructor for child class, should not be used to initialise Neural Network
//</summary>
NeuralNetwork::NeuralNetwork(){};
/// <summary>
/// Initializes a new instance of the <see cref="NeuralNetwork" /> class.
/// </summary>
/// <param name="neuronsCountPerLayer">The hidden layers.</param>
/// <param name="layercount">The hidden layercount.</param>
/// <param name="initType">Type of the initialize.</param>
/// <param name="functionType">Type of the function.</param>
NeuralNetwork::NeuralNetwork(int *neuronsCountPerLayer, int layercount, WeightInitEnum initType, FunctionEnum functionType) {
	init();

	if (layercount < 2)
		throw std::runtime_error("We need at least 2 Layer (input and output)");
	for (int i = 0; i < LayersCount; ++i) {
		if (neuronsCountPerLayer[i] < 1)
			throw std::runtime_error("Layer need at least 1 neuron");
	}

	this->FunctionType = functionType;

	LayersCount = layercount - 1; //Input Layer hat nicht wirklich neuronn also den nicht mitzählen

	Layers = new Layer[LayersCount]();
	for (int i = 0; i < LayersCount; ++i) {
		Layers[i].Init(neuronsCountPerLayer[i], neuronsCountPerLayer[i + 1]);
		TotalNeuronCount += neuronsCountPerLayer[i + 1];
	}

	InitWeights(initType);
}



/// <summary>
/// Operators the specified net.
/// </summary>
/// <param name="net">The net.</param>
/// <returns></returns>
bool NeuralNetwork::operator < (const NeuralNetwork& net) const {
	return (this->MeanSquareError < net.MeanSquareError);
}

/// <summary>
/// Frees the memory.
/// </summary>
void NeuralNetwork::freeMem() {
	delete [] Layers;
}

/// <summary>
/// Initializes this instance.
/// </summary>
void NeuralNetwork::init() {
	//Use random_device to generate a seed for Mersenne twister engine.
	std::random_device rd;
	// Use Mersenne twister engine to generate pseudo-random numbers.
	generator.seed(rd());

	LayersCount = 0;
	MeanSquareError = 0;

	WeightInitType = WeightInitEnum::NONE;
	FunctionType = FunctionEnum::LINEAR;
	TotalNeuronCount = 0;
	Layers = nullptr;
}

//ersthaft keine delegation constructor ?! todo: gucken ob das g++ kann
//NeuralNetwork::NeuralNetwork(int *layers,int layercount,WeightInitEnum initType,FunctionEnum functionType)
////	:NeuralNetwork()
//{
//	Init(layers,layercount,initType,functionType);
//}

/// <summary>
/// Finalizes an instance of the <see cref="NeuralNetwork"/> class.
/// </summary>
NeuralNetwork::~NeuralNetwork() {
	freeMem();
}

/// <summary>
/// Copies the specified that.
/// </summary>
/// <param name="that">The that.</param>
void NeuralNetwork::copy(const NeuralNetwork &that) {
	this->LayersCount = that.LayersCount;
	this->MeanSquareError = that.MeanSquareError;
	this->TotalNeuronCount = that.TotalNeuronCount;
	this->WeightInitType = that.WeightInitType;
	this->FunctionType = that.FunctionType;

	this->Layers = new Layer[LayersCount];

	for (int i = 0; i < LayersCount; i++)
		this->Layers[i] = that.Layers[i];
}

/// <summary>
/// Initializes a new instance of the <see cref="NeuralNetwork"/> class.
/// </summary>
/// <param name="that">The that.</param>
NeuralNetwork::NeuralNetwork(const NeuralNetwork &that)
// :NeuralNetwork() funktioniert unter g++ jedoch bekommt der windows compiler das leider nicht hin deshalb workaround mit init()
{
	init();
	copy(that);
}

/// <summary>
/// Operator=s the specified that.
/// </summary>
/// <param name="that">The that.</param>
/// <returns></returns>
NeuralNetwork& NeuralNetwork::operator= (const NeuralNetwork &that) {
	if (&that != this) {
		freeMem();
		init();
		copy(that);
	}
	return *this;
}

/// <summary>
/// Calculates the mse.
/// </summary>
/// <param name="data">The data.</param>
void NeuralNetwork::CalculateMSE(const DataContainer& data) {
	MeanSquareError = 0;

	for (int j = 0; j < data.DataCount; ++j) {
		Propagate(data.DataInput[j]);

		for (int i = 0; i < Layers[LayersCount - 1].NeuronCount; ++i) {
			double Output = Layers[LayersCount - 1].Neurons[i].Output;
			MeanSquareError += (data.DataOutput[j][i] - Output) * (data.DataOutput[j][i] - Output);
		}
	}

	MeanSquareError /= data.DataCount;
}

/// <summary>
/// Generates the random weight.
/// </summary>
/// <param name="inputVectorCountWithBias">The input vector count with bias.</param>
/// <returns></returns>
double NeuralNetwork::GenerateRandomWeight(const int inputVectorCountWithBias) {
	switch (WeightInitType) {
	case WeightInitEnum::NORMAL5: {
		//std::normal_distribution<> dist(0,2);
		std::normal_distribution<double> dist(0, 5);
		return dist(generator);
	}
	case WeightInitEnum::UNIFORM5: {
		std::uniform_real_distribution<double> dist(-5.0, 5.0);
		return dist(generator);
	}
	case WeightInitEnum::UNIFORM: {
		std::uniform_real_distribution<double> dist(-1.0, 1.0);
		return dist(generator);
	}
	case WeightInitEnum::LECUN: {
		double sigma = pow(inputVectorCountWithBias, -0.5);
		std::normal_distribution<double> dist(0.0, sigma);
		return dist(generator);
	}
	case WeightInitEnum::NORMAL0: {
		std::normal_distribution<double> dist(0, 0.01);
		return dist(generator);
	}
	case WeightInitEnum::NONE:
		return 0;
	case  WeightInitEnum::DEBUGONE:
		return 1;
	}

	return 0;
}

/// <summary>
/// Initializes the weights.
/// </summary>
/// <param name="initType">Type of the initialize.</param>
void NeuralNetwork::InitWeights(WeightInitEnum initType) {
	this->WeightInitType = initType;

	int i, j;
	for (i = (LayersCount - 1); i >= 0; i--) {
		Layer* layer = &Layers[i];

		for (j = 0; j < layer->NeuronCount; ++j) {
			Neuron* neuron = &layer->Neurons[j];

			for (int k = 0; k < layer->InputValuesCount + 1; ++k) { //+1 for Bias
				neuron->Weights[k] = GenerateRandomWeight(layer->InputValuesCountWithBias);
			}
		}
	}
}


/// <summary>
/// Saves the weights.
/// </summary>
/// <param name="file">The file.</param>
void NeuralNetwork::SaveWeights(const std::string file) {
	int i, j;

	std::ofstream myfile;
	myfile.open(file);

	if (!myfile) {
		std::string buf("Could not open file");
		buf.append(file);
		throw std::runtime_error(buf);
	}

	for (i = 0; i < LayersCount; ++i) {
		Layer* layer = &Layers[i];

		for (j = 0; j < layer->NeuronCount; ++j) {
			Neuron* neuron = &layer->Neurons[j];

			for (int k = 0; k < layer->InputValuesCountWithBias; ++k) {
				myfile << neuron->Weights[k] << "\n";
			}
		}
	}

	myfile.close();
}

/// <summary>
/// Loads the weights.
/// </summary>
/// <param name="file">The file.</param>
void NeuralNetwork::LoadWeights(const std::string file) {
	int i, j;
	std::ifstream iFile(file);
	std::string line;

	if (!iFile) {
		std::string buf("Could not open file");
		buf.append(file);
		throw std::runtime_error(buf);
	}

	for (i = 0; i < LayersCount; ++i) {
		Layer* layer = &Layers[i];

		for (j = 0; j < layer->NeuronCount; ++j) {
			Neuron* neuron = &layer->Neurons[j];

			for (int k = 0; k < layer->InputValuesCountWithBias; ++k) { //+1 for Bias
				getline(iFile, line);
				neuron->Weights[k] = atof(line.c_str());
			}
		}
	}

	iFile.close();
}

/// <summary>
/// Propagates the specified input.
/// </summary>
/// <param name="input">The input.</param>
void NeuralNetwork::Propagate(const double *input) {
	for (int i = 0; i < Layers[0].InputValuesCount; ++i) {
		//inputlayer hat keine neuronen daher daten in input vektor des ersten layers mit Neuronen übernehmen
		Layers[0].InputValues[i] = input[i];
	}

	int k, j;
	double net;
	Neuron *neuron;

	for (int i = 0; i < LayersCount; ++i) {
		for (k = 0; k < Layers[i].NeuronCount; ++k) {
			neuron = &Layers[i].Neurons[k];
			net = 0;

			//summiere gewichte * output des vorherigen layer auf => net
			//unrolling des loops beschleunigte das propagieren um Faktor ~2 zumindestens beim windows compiler
			/*for(j=0;j<Layers[i].InputValuesCount;++j)
			{
			net+=neuron->Weights[j] * Layers[i].InputValues[j];
			}*/

			//unrolling
			j = Layers[i].InputValuesCount & 3; //das gleiche wie module 4

			//behandle Neuronen die nicht in der nachfolgenden 4er Schritt loop behandelt werden können
			switch (j) {
			case 3:
				net += neuron->Weights[2] * Layers[i].InputValues[2];

			case 2:
				net += neuron->Weights[1] * Layers[i].InputValues[1];

			case 1:
				net += neuron->Weights[0] * Layers[i].InputValues[0];
			case 0:
				break;
			}

			for (; j != Layers[i].InputValuesCount; j += 4) {
				net +=
				    neuron->Weights[j] * Layers[i].InputValues[j] +
				    neuron->Weights[j + 1] * Layers[i].InputValues[j + 1] +
				    neuron->Weights[j + 2] * Layers[i].InputValues[j + 2] +
				    neuron->Weights[j + 3] * Layers[i].InputValues[j + 3];
			}

			net += neuron->Weights[Layers[i].InputValuesCount] /** neuron->Bias*/;
			//unrolling ende

			//aktivierungsfunktion f_net(net) auf net anwenden und in ausgabe des neurons speichern
			neuron->Output = ActivationFunction(FunctionType, net);

			//ausgaben des layers als input des nächsten layers verwenden
			if (i < LayersCount - 1) //letzter layer nicht weiter nach vorne kopieren
				Layers[i + 1].InputValues[k] = neuron->Output;
		}
	}
}


int Binary(double x) {
	std::default_random_engine generator;
	generator.seed(time(NULL));
	std::uniform_real_distribution<double> dist(0.0, 1.0);
	return x >= dist(generator) && x != 0.0;
}



}
