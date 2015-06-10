#include "DBNLayer.h"
#include "DBNNeuron.h"

namespace NNTLib
{
	/**
	 * @brief Frees Memory used by Layer
	 * @details The Neurons of the Layer get freed
	 */
	void DBNLayer::freeMem()
	{
		
		delete [] Neurons;
	}
	/**
	 * @brief Initiliases Layer
	 * @details Simple Initialisation without parameters sets everything to 0
	 */
	void DBNLayer::init()
	{
		InputValuesCount=0;
		InputValuesCountWithBias=0;
		NeuronCount=0;
		Neurons = nullptr;
		InputValues= nullptr;
		SumDeltaErrWeights= nullptr;
	}
	/**
	 * @brief Initialises Layer with parameters
	 * @details Initialises the Layer with Neurons, the number of inputvalues, the Deltas for errors in the weighst.
	 * Takes Care of building the weights Between Layers and that the Bias has no input weights
	 * 
	 * @param inputsize Number of Inputs on the Layer
	 * @param neuronCount Number of Neurons on the Layer
	 */
	void DBNLayer::Init(int inputsize, int neuronCount)
	{
		NeuronCount=(unsigned long)neuronCount;
		InputValuesCount=inputsize;
		Neurons=new DBNNeuron[NeuronCount]();
		InputValues=new double[InputValuesCount]();
		SumDeltaErrWeights = new double[inputsize]();
		if(inputsize!=0)
			InputValuesCountWithBias= InputValuesCount+1;

		for(int i=0;i<NeuronCount-1;++i)
		{
			Neurons[i].Init(InputValuesCountWithBias);
		}
		Neurons[NeuronCount-1].Init(InputValuesCount);//bias hat keine Eingabegewichte
	}
	/**
	 * @brief Creates Forwardweights
	 * @details Sets Pointers to the weights from the layer above, making it easier to access them from the down Layer
	 * 
	 * @param Neuronsdown Number of Neurons on the down Layer
	 * @param Layerup Pointer to the above Layer
	 */
	void DBNLayer::Forwardweightsinit(int Neuronsdown, DBNLayer* Layerup)
	{
		for (int i=0;i<=Neuronsdown;i++){
			Neurons[i].ForwardWeightCount=Layerup->NeuronCount-1;
			Neurons[i].ForwardWeights= new double*[Neurons[i].ForwardWeightCount];
			for (int j=0; j<Layerup->NeuronCount-1;j++){ //-1 da keine Gewichte zum Bias
				Neurons[i].ForwardWeights[j]=&Layerup->Neurons[j].Weights[i];
			}
		}

	}
	/**
	 * @brief Copies one layer to another
	 * @details Initialises a new Layer with another one
	 * 
	 * @param that Layer that gets copied
	 */
	void DBNLayer::copy(const DBNLayer &that)
	{
		this->InputValuesCount = that.InputValuesCount;
		this->InputValuesCountWithBias = that.InputValuesCountWithBias;
		this->NeuronCount = that.NeuronCount;

		this->Neurons = new DBNNeuron[this->NeuronCount];
		this->InputValues = new double[this->InputValuesCount];
		this->SumDeltaErrWeights = new double[this->InputValuesCount];

		for(int i=0;i<this->NeuronCount;i++)
			this->Neurons[i] = that.Neurons[i];

		for(int i=0;i<InputValuesCount;i++)
		{
			this->InputValues[i]=that.InputValues[i];
			this->SumDeltaErrWeights[i]=that.SumDeltaErrWeights[i];
		}
	}
	/**
	 * @brief copy constructor
	 * @details copy constructor
	 * 
	 * @param that Layer to copy
	 */
	DBNLayer::DBNLayer(const DBNLayer &that)
	{
		init();
		copy(that);
	}
	/**
	 * @brief overloaded =
	 * @details Copies one layer into another
	 * 
	 * @param that layer to copy
	 * @return new layer with copied values
	 */
	DBNLayer& DBNLayer::operator= (const DBNLayer &that)
	{
		if (&that != this) {
			freeMem();
			init();
			copy(that);
		}
		return *this;
	}

	/// <summary>
	/// Initializes a new instance of the <see cref="DBNLayer"/> class.
	/// </summary>
	DBNLayer::DBNLayer()
	{
		init();
	}

	/// <summary>
	/// Finalizes an instance of the <see cref="DBNLayer"/> class.
	/// </summary>
	DBNLayer::~DBNLayer()
	{
		freeMem();
	}
}