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
		int ForwardWeightCount;
		//Aktivierungswahrscheinlichkeit bei stochastisch binären Neuronen
		double p;
		//Gewichte nach vorne um von allen Layern auf die Gewichte zugreifen zu können
		double **ForwardWeights;

		DBNNeuron();
		~DBNNeuron();
		DBNNeuron(const DBNNeuron &that);
		DBNNeuron& operator= (const DBNNeuron &that);

		//Initialisiert Gewichte des Bias für sichtbaren Layer in einer Boltzmann Maschine
		void InitBias(const DataContainer *container);

		void Init(int weightCount);
	};
}
#endif