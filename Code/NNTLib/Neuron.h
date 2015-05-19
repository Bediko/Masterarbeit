#ifndef Neuron_H
#define Neuron_H

#include "Functions.h"
#include "DataContainer.h"
namespace NNTLib
{
	class Neuron
	{
	protected:
		void copy(const Neuron &that);
		void init();
		void freeMem();
	public:
		/// <summary>
		/// The weight count (Anzahl eingehende Gewichte (mit Schwellenwert)
		/// </summary>
		int WeightCount;

		/// <summary>
		/// The output (Ausgabe des Neurons)
		/// </summary>
		double Output;
		/// <summary>
		/// The weights (Collection aller Gewichte inklusive Schwellenwert Gewicht)
		/// </summary>
		double *Weights;
		/// <summary>
		/// The last delta weights (letzte Gewichtändeurng wird für Momentum Verfahren benötigt)
		/// </summary>
		double *LastDeltaWeights;
		/// <summary>
		/// The delta weights (Summe aller Gewichtsänderungen innerhalb einer Batchsize, wird bei online training nicht verwendet)
		/// </summary>
		double *DeltaWeights;
		//bias einzelnt halten (für evtl. erweiterung von backpropagation mit bias anpassung siehe Lecun Paper)
		double Bias;


		Neuron();
		~Neuron();
		Neuron(const Neuron &that);
		Neuron& operator= (const Neuron &that);

		void Init(int weightCount);
	};
}
#endif
