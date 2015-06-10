#ifndef DBNNeuron_H
#define DBNNeuron_H

#include "Functions.h"
#include "DataContainer.h"
#include "Neuron.h"
namespace NNTLib
{
	class DBNNeuron : public Neuron
	{
	protected:
		void copy(const DBNNeuron &that);
		void init();
		void freeMem();
	public:
		/**
		 * @brief Number of Forwardweights
		 */
		int ForwardWeightCount;
		/**
		 * @brief propability to turn on
		 */
		double p;
		/**
		 * @brief Pointer to weights in the layer above
		 */
		double **ForwardWeights;

		DBNNeuron();
		~DBNNeuron();
		DBNNeuron(const DBNNeuron &that);
		DBNNeuron& operator= (const DBNNeuron &that);
		void InitBias(const DataContainer *container);

		void Init(int weightCount);
	};
}
#endif