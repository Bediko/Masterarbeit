#ifndef Neuron_H
#define Neuron_H

#include "Functions.h"
#include "DataContainer.h"
namespace NNTLib
{
	class Neuron
	{
	private:
		void copy(const Neuron &that);
		void init();
		void freeMem();
	public:
		/// <summary>
		/// The weight count (Anzahl eingehende Gewichte (mit Schwellenwert)
		/// </summary>
		int WeightCount;
		int ForwardWeightCount;
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

		//Gewichte nach vorne um von allen Layern auf die Gewichte zugreifen zu können
		double **ForwardWeights;

		Neuron();
		~Neuron();
		Neuron(const Neuron &that);
		Neuron& operator= (const Neuron &that);

		//Initialisiert Gewichte des Bias für sichtbaren Layer in einer Boltzmann Maschine
		void InitBias(const DataContainer *container);

		void Init(int weightCount);
	};
}
#endif
