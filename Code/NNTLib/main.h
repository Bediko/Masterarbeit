#ifndef MAIN_H
#define MAIN_H
#include <cmath>
#include "Helper.h"


#include "NeuralNetworkConfig.h"
#include "GeneticAlgorithmConfig.h"
#include "BackpropagationConfig.h"
#include "ContrastiveDivergenceConfig.h"

class Layer;
class Neuron;
void SimpleXORExampleBackpropagation();
void SimpleXORExampleGenetic();
void SimpleMNISTExampleBackpropagation();
void SimpleMNISTExampleGenetic();
void DisplaySingleMNISTChar(const NNTLib::DataContainer *dataContainermnistTest, int index);

const std::string LINE = "-----------------------------------------\n";
const std::string ARGUMENT_MESSAGE ="Argumente:\n"
	"	-N [File]	Lade Netzwerk Konfigurationsdatei. \n "
	"	-D [File]	Lade Daten, ohne -K [Value] werden alle Daten zum \n"
	"			Training verwendet. \n "
	"	-TD [File]	Lade zus\x84tzliche Testdaten. \n "
	"	-B [File]	Lade Konfigurationsdatei f\x81r Backpropagation. \n "
	"	-G [File]	Lade Konfigurationsdatei f\x81r genetisches\n"
	"			Training. \n "
	"	-CD [File]	Lade Konfigurationsdatei f\x81r Contrastive Divergence. \n"
	"	-LW [File]	Lade Gewichte, diese werden vor dem Training\n"
	"			und Test geladen. \n "
	"	-SW [File]	Speicherung der Gewichte nach dem Training, die\n"
	"			Variablen %N (%B|%G) %D in [File] werden mit dem \n "
	"			jeweiligen Dateinamen des Parameters ersetzte,\n"
	"			%K wird mit dem Iterationsschritt 0...K-1 ersetzt \n "
	"	-SR [File]	Speichern der Ergebnisse des jeweiligen Trainings\n"
	"			(Iterationsschritt MSE ZEIT)\n "
	"			Varibalen %N (%B|%G) %D %K werden in [File] ersetzt siehe -SW\n "
	"	-C [Value]	Teste Daten, ab einer Abweichung von [Value] wird das\n"
	"			Ergebnis als Fehler gez\x84hlt. Bei mehr als einem\n"
	"			Ausgabe-Neuron wird Value ignoriert.\n "
	"	-K [Value]	Kreuzvalidierung, teilt Daten in K teile und trainiert\n"
	"			das Netzwerk mit jeweils K-1 teilen und testet diese\n"
	"			anschlieﬂend mit dem k-ten Teil (falls Daten nicht \n "
	"			durch k teilbar werden die restlichen Datens\x84tze\n"
	"			zum Test hinzugez\x84hlt)\n "
	"	-O [Value]	Offset in ms. Bei Speicherung der Ergebnisse \n"
	"			werden die Zeiten um den Offset versetzt.\n"
	"	-I [Value]	Anzahl Wiederholungen. Zus\x84tzlich werden Mittelwert\n"
	"			und Streuung berechnet.\n";

#endif
