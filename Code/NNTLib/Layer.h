#ifndef Layer_H
#define Layer_H

#include "Neuron.h"
namespace NNTLib
{
	class Layer
	{
	private:
		void copy(const Layer &that);
		void init();
		void freeMem();
	public:
		/// <summary>
		/// The input values count
		/// </summary>
		int InputValuesCount;
		/// <summary>
		/// The input values count with bias
		/// </summary>
		int InputValuesCountWithBias;
		/// <summary>
		/// The neuron count
		/// </summary>
		int NeuronCount;
		/// <summary>
		/// The neurons (Collection aller Neuronen auf dem Layer)
		/// </summary>
		Neuron *Neurons;
		/// <summary>
		/// The input values (Collection mit Ausgaben des darunter liegenden Layers Li-1)
		/// </summary>
		double *InputValues;
		/// <summary>
		/// The sum (delta * error * weights)
		/// </summary>
		double *SumDeltaErrWeights;

		void Init(int inputsize, int neuronCount);
		void Init(int inputsize, int neuronCount, int dbn);
		void Forwardweightsinit(int inputsize, Layer* Layerup, int dbn);

		Layer();

		~Layer();
		Layer(const Layer &that);
		Layer& operator= (const Layer &that);
	};
}
#endif