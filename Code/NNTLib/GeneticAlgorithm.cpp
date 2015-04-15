#include "GeneticAlgorithm.h"
namespace NNTLib
{
	/// <summary>
	/// Initializes a new instance of the <see cref="GeneticAlgorithm"/> class.
	/// </summary>
	/// <param name="currentPopulation">The current population.</param>
	/// <param name="populationSize">Size of the population.</param>
	/// <summary>
	/// Initializes a new instance of the <see cref="GeneticAlgorithm"/> class.
	/// </summary>
	/// <param name="currentPopulation">The current population.</param>
	/// <param name="populationsize">The populationsize.</param>
	GeneticAlgorithm::GeneticAlgorithm(NeuralNetwork **currentpopulation,int populationsize)
	{
		if(populationsize <= 0)
			throw std::runtime_error("populationSize muss größer gleich 1 sein!");

		rouletteSum=0;
		this->currentPopulation = currentpopulation;

		//Use random_device to generate a seed for Mersenne twister engine.
		std::random_device rd;

		// Use Mersenne twister engine to generate pseudo-random numbers.
		generator.seed(rd());

		this->populationSize = populationsize;

		nextPopulation = new NeuralNetwork*[populationSize];

		for(int k =0;k<populationSize;++k)
		{
			//kopie erstellen für die Layers zwar werden die gewichte auch übernommen
			//jedoch werden diese bei crossover sowieso überschrieben
			nextPopulation[k] = new NeuralNetwork(*currentPopulation[k]);
		}
	}

	/// <summary>
	/// Finalizes an instance of the <see cref="GeneticAlgorithm"/> class.
	/// </summary>
	GeneticAlgorithm::~GeneticAlgorithm()
	{
		if(nextPopulation)
		{
			for(int k =0;k<populationSize;++k)
			{
				delete nextPopulation[k];
			}
			delete [] nextPopulation;
		}
	}

	/// <summary>
	/// Roulettes the index fittnes based.
	/// </summary>
	/// <param name="skip">The skip.</param>
	/// <returns></returns>
	int GeneticAlgorithm::rouletteIndexFittnesBased(int skip)
	{
		//Siehe Funktion rouletteIndexRankBased für Erklärung was hier passiert
		std::uniform_real_distribution<double> dist(0, rouletteSum);
		double r = dist(generator);
		double offset =0;
		for (int pick = 0; pick < populationSize-skip; pick++) {
			offset += currentPopulation[pick]->MeanSquareError;
			if (r < offset) {
				return (populationSize-1) - pick-skip;
			}
		}

		//darf hier nicht hin kommen
		return 0;
	}

	/// <summary>
	/// Calculates the roulette sum.
	/// </summary>
	/// <param name="rouletteType">Type of the roulette.</param>
	/// <param name="skip">The skip.</param>
	void GeneticAlgorithm::calculateRouletteSum(RouletteEnum rouletteType,int skip)
	{
		rouletteSum=0;
		switch (rouletteType)
		{
		case NNTLib::RouletteEnum::FITTNESSBASED:
			for(int i =0;i<populationSize-skip;i++)
			{
				rouletteSum+=currentPopulation[i]->MeanSquareError;
			}
			break;
		case NNTLib::RouletteEnum::INDEXBASED:
			rouletteSum = ((populationSize-skip)*((populationSize-skip)+1))/2;//summe über Anzahl Elemente bei population = 10 => 55 (Gaußsche Summenformel)
			break;
		default:
			break;
		}
	}

	/// <summary>
	/// Roulettes the index rank based.
	/// </summary>
	/// <param name="skip">The skip. gibt Anzahl der nicht einzubeziehenden Items an (zb skip=1 wenn man den schlechtesten chromosom nicht einbeziehen will)</param>
	/// <returns></returns>
	int GeneticAlgorithm::rouletteIndexRankBased(int skip)
	{
		std::uniform_real_distribution<double> dist(0, rouletteSum);
		double r = dist(generator);
		int offset =0;

		for (int pick = 0; pick < populationSize-skip; pick++) {
			offset += (pick+1);
			if (r < offset) {
				//pick entspricht index
				//da der MeanSquareError je kleiner umso besser ist müsste eigentlich das inverse MeanSquareError benutzt werden das wird
				//gespart indem hier einfach der inverse Index gewählt wird
				return (populationSize-1) - pick -skip;
			}
		}

		//darf hier nicht hin kommen
		return 0;
	}

	//sortieren std::sort aus algroithm war bei geringer Anzahl Population und großen Objekten durch das kopieren der Element zu
	//langsam bzw unklar wie optimal zu verwenden, deshalb ganz simpel nach einander jedes neu reinkommende element an seine richtige stelle
	//plazieren (nur tausch operationen)
	/// <summary>
	/// Sorts the single element.
	/// </summary>
	/// <param name="index">The index.</param>
	void GeneticAlgorithm::sortSingleElement(int index)
	{
		/*
		Elemente kommen nacheinander in die liste rein angefangen bei index 0
		Schritt 1			Schritt 2
		#index|MSE			#index|MSE
		0|3					0|tmp
		1|4					1|3
		2|2<- new value		2|4
		3|0					3|0
		new index = 0		tmp = 2;
		*/

		for(int i=0;i<index;i++)
		{
			//suche index mit schlechterem mse
			if(currentPopulation[index]->MeanSquareError < currentPopulation[i]->MeanSquareError)
			{
				NeuralNetwork * tmp = currentPopulation[index];
				//verschiebe alle werte um ein index nach unten
				for(int j=index;j>=i;j--)
				{
					currentPopulation[j]=currentPopulation[j-1];
				}
				//plaziere neues element an richtiger stelle
				currentPopulation[i]=tmp;
				break;
			}
		}
	}

	/// <summary>
	/// Trains the specified data container.
	/// </summary>
	/// <param name="dataContainer">The data container.</param>
	/// <param name="maxLoopCount">The maximum loop count.</param>
	/// <param name="errorThreshold">The error threshold.</param>
	/// <param name="mutateType">Type of the mutate.</param>
	/// <param name="crossoverType">Type of the crossover.</param>
	/// <param name="rouletteType">Type of the roulette.</param>
	/// <param name="eltismCount">The eltism count.</param>
	/// <param name="mutationProbability">The mutation probability.</param>
	/// <param name="crossoverProbability">The crossover probability.</param>
	/// <param name="mutateNodeCount">The mutate node count.</param>
	void GeneticAlgorithm::Train(const DataContainer &dataContainer,int maxLoopCount,double errorThreshold,MutateEnum mutateType,CrossoverEnum crossoverType, RouletteEnum rouletteType,int eltismCount,double mutationProbability,double crossoverProbability,int mutateNodeCount)
	{
		if(mutateNodeCount > currentPopulation[0]->TotalNeuronCount)
			throw std::runtime_error("mutateNodeCount > TotalNeuronCount !");

		if(eltismCount > populationSize)
			throw std::runtime_error("eltismCount > populationSize !");

		if(eltismCount >= (populationSize-1))
		{
			trainSteadyState(dataContainer,maxLoopCount,errorThreshold,mutateType,crossoverType,rouletteType,mutationProbability,crossoverProbability,mutateNodeCount);
		}
		else
		{
			trainPopulation(dataContainer,maxLoopCount,errorThreshold,mutateType,crossoverType,rouletteType,eltismCount,mutationProbability,crossoverProbability,mutateNodeCount);
		}
	}

	/// <summary>
	/// Trains the population.
	/// </summary>
	/// <param name="dataContainer">The data container.</param>
	/// <param name="maxLoopCount">The maximum loop count.</param>
	/// <param name="errorThreshold">The error threshold.</param>
	/// <param name="mutateType">Type of the mutate.</param>
	/// <param name="crossoverType">Type of the crossover.</param>
	/// <param name="rouletteType">Type of the roulette.</param>
	/// <param name="eltismCount">The eltism count.</param>
	/// <param name="mutationProbability">The mutation probability.</param>
	/// <param name="crossoverProbability">The crossover probability.</param>
	/// <param name="mutateNodeCount">The mutate node count.</param>
	void GeneticAlgorithm::trainPopulation(const DataContainer & dataContainer,int maxLoopCount,double errorThreshold,MutateEnum mutateType,CrossoverEnum crossoverType, RouletteEnum rouletteType,int eltismCount,double mutationProbability,double crossoverProbability,int mutateNodeCount)
	{
		int populationIterationSize = populationSize - eltismCount;

		int indexFirstParent=0;
		int indexSecondParent=0;

		initMeasureResult(maxLoopCount);

		//Mean Square Errors berechnen und absteigend nach mse sortieren
		for(int k =0;k<populationSize;++k)
		{
			currentPopulation[k]->CalculateMSE(dataContainer);
			sortSingleElement(k);

			nextPopulation[k]->MeanSquareError = currentPopulation[k]->MeanSquareError;
		}

		std::uniform_real_distribution<double> dist(0.0, 1.0);

		//für zufällige auswahl eines Neurons z.B für MutateEnum::MUTATE_NODES
		std::uniform_int_distribution<int> distNeuronIndex(0, currentPopulation[0]->TotalNeuronCount);

		//das sort kopiert zuviel rum deshalb ein simples verfahren siehe SortSingleElement
		//std::sort(currentPopulation,currentPopulation+populationSize);

		//Anfangszeitpunkt holen
		unsigned long long time64= GetTimeMs64();

		int firstCrosspoint = 0;
		int secondCrosspoint = 0;
		if(crossoverType == CrossoverEnum::CROSSOVER_ONEPOINT || crossoverType == CrossoverEnum::CROSSOVER_TWOPOINT)
			firstCrosspoint = distNeuronIndex(generator);

		if(crossoverType == CrossoverEnum::CROSSOVER_TWOPOINT)
		{
			secondCrosspoint = distNeuronIndex(generator);

			while(firstCrosspoint == secondCrosspoint)
				secondCrosspoint = distNeuronIndex(generator);

			if(secondCrosspoint < firstCrosspoint)
			{
				int tmp= secondCrosspoint;
				secondCrosspoint = firstCrosspoint;
				firstCrosspoint = tmp;
			}
		}

		int loopIndex;
		for(loopIndex=0;loopIndex<maxLoopCount;loopIndex++)
		{
			calculateRouletteSum(rouletteType);
			//bei eltismCount anfangen die restlichen gewichte wurden 1:1 übernommen
			for(int i=0;i< populationIterationSize;i++)
			{
				//zwei Netze wählen
				if(rouletteType == RouletteEnum::FITTNESSBASED)
				{
					indexFirstParent = rouletteIndexFittnesBased();
					indexSecondParent = rouletteIndexFittnesBased();
				}
				else if(rouletteType == RouletteEnum::INDEXBASED)
				{
					indexFirstParent = rouletteIndexRankBased();
					indexSecondParent = rouletteIndexRankBased();
				}

				int indexFirstNext = eltismCount+ i;
				//der zweite bekommt die inversen Gene wie der erste
				//int indexSecondNext = eltismCount+i+populationIterationSize/2;

				int neuronCounter=0;
				//da alle netze gleich sind wird der erste zum iterieren verwendet
				for (int k = 0; k < currentPopulation[0]->LayersCount; ++k)
				{
					for (int l = 0; l < currentPopulation[0]->Layers[k].NeuronCount; ++l)//neuronen count per Layer
					{
						neuronCounter++;
						double *weightsCurrentFirst =currentPopulation[indexFirstParent]->Layers[k].Neurons[l].Weights;
						double *weightsCurrentSecond =currentPopulation[indexSecondParent]->Layers[k].Neurons[l].Weights;
						double *weightsNextFirst =nextPopulation[indexFirstNext]->Layers[k].Neurons[l].Weights;
						//double *weightsNextSecond =nextPopulation[indexSecondNext]->Layers[k].Neurons[l].Weights;
						int inputCount = currentPopulation[indexFirstParent]->Layers[k].InputValuesCountWithBias;

						if(crossoverType == CrossoverEnum::CROSSOVER_ONEPOINT)
						{
							if(neuronCounter <= firstCrosspoint)
							{
								for(int j=0;j<inputCount;++j)//für jedes eingehende gewicht incl. Bias
								{
									weightsNextFirst[j] = weightsCurrentFirst[j];
								}
							}
							else
							{
								for(int j=0;j<inputCount;++j)//für jedes eingehende gewicht incl. Bias
								{
									weightsNextFirst[j] = weightsCurrentSecond[j];
								}
							}
						}
						else if(crossoverType == CrossoverEnum::CROSSOVER_TWOPOINT)
						{
							if(neuronCounter <= firstCrosspoint || neuronCounter > secondCrosspoint)
							{
								for(int j=0;j<inputCount;++j)//für jedes eingehende gewicht incl. Bias
								{
									weightsNextFirst[j] = weightsCurrentFirst[j];
								}
							}
							else
							{
								for(int j=0;j<inputCount;++j)//für jedes eingehende gewicht incl. Bias
								{
									weightsNextFirst[j] = weightsCurrentSecond[j];
								}
							}
						}
						else if(crossoverType == CrossoverEnum::CROSSOVER_NODES)
						{
							if(dist(generator) < crossoverProbability)//bei einer crossoverProbability von 0 werden diese einfach übernommen
							{
								for(int j=0;j<inputCount;++j)//für jedes eingehende gewicht incl. Bias
								{
									weightsNextFirst[j] = weightsCurrentFirst[j];
									//weightsNextSecond[j] = weightsCurrentSecond[j];
								}
							}
							else
							{
								for(int j=0;j<inputCount;++j)//für jedes eingehende gewicht incl. Bias
								{
									weightsNextFirst[j] = weightsCurrentSecond[j];
									//weightsNextSecond[j] = weightsCurrentFirst[j];
								}
							}
						}
						else if(crossoverType == CrossoverEnum::CROSSOVER_WEIGHTS)
						{
							for(int j=0;j<inputCount;++j)//für jedes eingehende gewicht incl. Bias
							{
								//rekombination
								if(dist(generator) < crossoverProbability)//bei einer crossoverProbability von 0 werden diese einfach übernommen
								{
									weightsNextFirst[j] = weightsCurrentFirst[j];
									//weightsNextSecond[j] = weightsCurrentSecond[j];
								}
								else
								{
									weightsNextFirst[j] = weightsCurrentSecond[j];
									//weightsNextSecond[j] = weightsCurrentFirst[j];
								}
							}
						}

						if(mutateType == MutateEnum::UNBIASED_MUTATA_WEIGHTS)
						{
							for(int j=0;j<inputCount;++j)//für jedes eingehende gewicht incl. Bias
							{
								//mutation
								if(dist(generator) < mutationProbability)
								{
									weightsNextFirst[j] =  currentPopulation[i]->GenerateRandomWeight(inputCount);// distWeights(engine);
									//weightsNextSecond[j] = currentPopulation[i]->GenerateRandomWeight(inputCount);
								}
							}
						}
						else if(mutateType == MutateEnum::BIAS_ED_MUTATE_WEIGHTS)
						{
							for(int j=0;j<inputCount;++j)//für jedes eingehende gewicht incl. Bias
							{
								//mutation
								if(dist(generator) < mutationProbability)
								{
									weightsNextFirst[j] += currentPopulation[i]->GenerateRandomWeight(inputCount);
									//weightsNextSecond[j] += currentPopulation[i]->GenerateRandomWeight(inputCount);
								}
							}
						}
					}
				}

				if(mutateType == MutateEnum::MUTATE_NODES)
				{
					int NeuronIndex = 0;

					int tmpNodeCounter = mutateNodeCount;

					while(tmpNodeCounter > 0)
					{
						tmpNodeCounter--;

						//unklar ob das gleiche Neuron in einer Iteration zweimal mutieren werden darf...ich entscheide mich einfach mal für ja
						int selectedNeuronenindex = distNeuronIndex(generator);

						int sumNeurons = 0;

						for(int n=0;n<currentPopulation[indexFirstParent]->LayersCount;n++)//Iteration durch alle Layer
						{
							sumNeurons+=currentPopulation[indexFirstParent]->Layers[n].NeuronCount;//summiere Anzahl Neuronen bis zum momentanten Layer auf

							if(selectedNeuronenindex < sumNeurons)
							{
								NeuronIndex = selectedNeuronenindex - (sumNeurons -currentPopulation[indexFirstParent]->Layers[n].NeuronCount);

								for(int j=0;j<currentPopulation[indexFirstParent]->Layers[n].InputValuesCountWithBias;++j)//für jedes eingehende gewicht incl. Bias
								{
									nextPopulation[indexFirstNext]->Layers[n].Neurons[NeuronIndex].Weights[j] += currentPopulation[i]->GenerateRandomWeight(currentPopulation[i]->Layers[n].InputValuesCountWithBias);
									//nextPopulation[indexSecondNext]->Layers[n].Neurons[NeuronIndex].Weights[j] += currentPopulation[i]->GenerateRandomWeight(currentPopulation[i]->Layers[n].InputValuesCountWithBias);
								}

								break; //nächstes Neuron
							}
						}
					}
				}
			}

			MeasureResult[loopIndex].ExecuteTime = GetTimeMs64() - time64;
			MeasureResult[loopIndex].MeanSquareError = currentPopulation[0]->MeanSquareError;//speichern nur den besten

			//Abbruch Kriterium prüfen
			if(MeasureResult[loopIndex].MeanSquareError <= errorThreshold)
			{
				break;
			}

			/*
			Bei Eltism werden am Anfang die besten Lösungen in die nächste Generation übernommen um zu vermeiden diese zu verlieren,
			anschließend werden die Restlichen Lösungen über die klassischen Auswahl Methoden ermittelt.
			*/
			double * weightsNext;
			double * weightsCurrent;
			for(int i=0;i< eltismCount;i++)
			{
				for (int k = 0; k < currentPopulation[0]->LayersCount; ++k)
				{
					for (int l = 0; l < currentPopulation[0]->Layers[k].NeuronCount; ++l)//neuronen count per Layer
					{
						//next ist hier die letzte population
						weightsNext = nextPopulation[i]->Layers[k].Neurons[l].Weights;
						weightsCurrent = currentPopulation[i]->Layers[k].Neurons[l].Weights;
						for(int j=0;j< currentPopulation[0]->Layers[k].InputValuesCountWithBias;++j)
						{
							weightsNext[j]=weightsCurrent[j];
						}
					}
				}

				nextPopulation[i]->MeanSquareError = currentPopulation[i]->MeanSquareError;
			}

			//current und next wechseln
			NeuralNetwork ** tmp = currentPopulation;
			currentPopulation = nextPopulation;
			nextPopulation = tmp;///da die gewichte sowieso ersetzt werden ist der inhalt egal bis auf mse das muss auf 0

			for(int i=eltismCount;i< populationSize;i++)
			{
				//berechne MSE für neue currentPopulation
				currentPopulation[i]->CalculateMSE(dataContainer);
				sortSingleElement(i);
				nextPopulation[i]->MeanSquareError = 0;//mse auf 0
			}

			//rouletteSum = 0;
		}

		MeasureFilledResultLenght = loopIndex;

		if(loopIndex % 2 == 1)
		{
			//wieder auf ursprüngliche adressen zeigen um nicht den falschen zu löschen
			NeuralNetwork ** tmp = currentPopulation;
			currentPopulation = nextPopulation;
			nextPopulation = tmp;
		}
	}

	/// <summary>
	/// Trains the state of the steady.
	/// </summary>
	/// <param name="dataContainer">The data container.</param>
	/// <param name="maxLoopCount">The maximum loop count.</param>
	/// <param name="errorThreshold">The error threshold.</param>
	/// <param name="mutateType">Type of the mutate.</param>
	/// <param name="crossoverType">Type of the crossover.</param>
	/// <param name="rouletteType">Type of the roulette.</param>
	/// <param name="mutationProbability">The mutation probability.</param>
	/// <param name="crossoverProbability">The crossover probability.</param>
	/// <param name="mutateNodeCount">The mutate node count.</param>
	void GeneticAlgorithm::trainSteadyState(const DataContainer & dataContainer,int maxLoopCount,double errorThreshold,MutateEnum mutateType,CrossoverEnum crossoverType, RouletteEnum rouletteType,double mutationProbability,double crossoverProbability,int mutateNodeCount)
	{
		int indexFirstParent=0;
		int indexSecondParent=0;
		int indexChild= populationSize-1; //bei steadystate fliegt immer der schlechteste

		initMeasureResult(maxLoopCount);

		//Mean Square Errors berechnen und absteigend nach mse sortieren
		for(int k =0;k<populationSize;++k)
		{
			currentPopulation[k]->CalculateMSE(dataContainer);
			sortSingleElement(k);
		}

		std::uniform_real_distribution<double> dist(0.0, 1.0);

		if(mutateNodeCount >currentPopulation[0]->TotalNeuronCount)
			mutateNodeCount = currentPopulation[0]->TotalNeuronCount;

		//für zufällige auswahl eines Neurons z.B für MutateEnum::MUTATE_NODES
		std::uniform_int_distribution<int> distNeuronIndex(0, currentPopulation[0]->TotalNeuronCount);

		//das sort kopiert zuviel rum deshalb ein simples verfahren siehe SortSingleElement
		//std::sort(currentPopulation,currentPopulation+populationSize);

		//Anfangszeitpunkt holen
		unsigned long long time64= GetTimeMs64();

		int firstCrosspoint = 0;
		int secondCrosspoint = 0;

		if(crossoverType == CrossoverEnum::CROSSOVER_ONEPOINT || crossoverType == CrossoverEnum::CROSSOVER_TWOPOINT)
			firstCrosspoint = distNeuronIndex(generator);

		if(crossoverType == CrossoverEnum::CROSSOVER_TWOPOINT)
		{
			secondCrosspoint = distNeuronIndex(generator);

			while(firstCrosspoint == secondCrosspoint)
				secondCrosspoint = distNeuronIndex(generator);

			if(secondCrosspoint < firstCrosspoint)
			{
				int tmp= secondCrosspoint;
				secondCrosspoint = firstCrosspoint;
				firstCrosspoint = tmp;
			}
		}

		int loopIndex;
		for(loopIndex=0;loopIndex<maxLoopCount;loopIndex++)
		{
			calculateRouletteSum(rouletteType,1);
			//zwei Netze wählen
			if(rouletteType == RouletteEnum::FITTNESSBASED)
			{
				indexFirstParent = rouletteIndexFittnesBased(1);
				indexSecondParent = rouletteIndexFittnesBased(1);
			}
			else if(rouletteType == RouletteEnum::INDEXBASED)
			{
				indexFirstParent = rouletteIndexRankBased(1);
				indexSecondParent = rouletteIndexRankBased(1);
			}

			int neuronCounter=0;
			//da alle netze gleich sind wird der erste zum iterieren verwendet
			for (int k = 0; k < currentPopulation[0]->LayersCount; ++k)
			{
				for (int l = 0; l < currentPopulation[0]->Layers[k].NeuronCount; ++l)//neuronen count per Layer
				{
					neuronCounter++;

					double *weightsCurrentFirst =currentPopulation[indexFirstParent]->Layers[k].Neurons[l].Weights;
					double *weightsCurrentSecond =currentPopulation[indexSecondParent]->Layers[k].Neurons[l].Weights;

					double *weightsNextFirst =currentPopulation[indexChild]->Layers[k].Neurons[l].Weights;

					int inputCount = currentPopulation[indexFirstParent]->Layers[k].InputValuesCountWithBias;

					if(crossoverType == CrossoverEnum::CROSSOVER_ONEPOINT)
					{
						if(neuronCounter <= firstCrosspoint)
						{
							for(int j=0;j<inputCount;++j)//für jedes eingehende gewicht incl. Bias
							{
								weightsNextFirst[j] = weightsCurrentFirst[j];
							}
						}
						else
						{
							for(int j=0;j<inputCount;++j)//für jedes eingehende gewicht incl. Bias
							{
								weightsNextFirst[j] = weightsCurrentSecond[j];
							}
						}
					}
					else if(crossoverType == CrossoverEnum::CROSSOVER_TWOPOINT)
					{
						if(neuronCounter <= firstCrosspoint || neuronCounter > secondCrosspoint)
						{
							for(int j=0;j<inputCount;++j)//für jedes eingehende gewicht incl. Bias
							{
								weightsNextFirst[j] = weightsCurrentFirst[j];
							}
						}
						else
						{
							for(int j=0;j<inputCount;++j)//für jedes eingehende gewicht incl. Bias
							{
								weightsNextFirst[j] = weightsCurrentSecond[j];
							}
						}
					}
					else if(crossoverType == CrossoverEnum::CROSSOVER_NODES)
					{
						if(dist(generator) < crossoverProbability)//bei einer crossoverProbability von 0 werden diese einfach übernommen
						{
							for(int j=0;j<inputCount;++j)//für jedes eingehende gewicht incl. Bias
							{
								weightsNextFirst[j] = weightsCurrentFirst[j];
							}
						}
						else
						{
							for(int j=0;j<inputCount;++j)//für jedes eingehende gewicht incl. Bias
							{
								weightsNextFirst[j] = weightsCurrentSecond[j];
							}
						}
					}
					else if(crossoverType == CrossoverEnum::CROSSOVER_WEIGHTS)
					{
						for(int j=0;j<inputCount;++j)//für jedes eingehende gewicht incl. Bias
						{
							//rekombination
							if(dist(generator) < crossoverProbability)//bei einer crossoverProbability von 0 werden diese einfach übernommen
							{
								weightsNextFirst[j] = weightsCurrentFirst[j];
							}
							else
							{
								weightsNextFirst[j] = weightsCurrentSecond[j];
							}
						}
					}

					if(mutateType == MutateEnum::UNBIASED_MUTATA_WEIGHTS)
					{
						for(int j=0;j<inputCount;++j)//für jedes eingehende gewicht incl. Bias
						{
							//mutation
							if(dist(generator) < mutationProbability)
							{
								weightsNextFirst[j] =  currentPopulation[indexChild]->GenerateRandomWeight(inputCount);// distWeights(engine);
							}
						}
					}
					else if(mutateType == MutateEnum::BIAS_ED_MUTATE_WEIGHTS)
					{
						for(int j=0;j<inputCount;++j)//für jedes eingehende gewicht incl. Bias
						{
							//mutation
							if(dist(generator) < mutationProbability)
							{
								weightsNextFirst[j] += currentPopulation[indexChild]->GenerateRandomWeight(inputCount);
							}
						}
					}
				}
			}

			if(mutateType == MutateEnum::MUTATE_NODES)
			{
				int NeuronIndex = 0;
				int tmpNodeCounter = mutateNodeCount;
				while(tmpNodeCounter > 0)
				{
					tmpNodeCounter--;
					//unklar ob das gleiche Neuron in einer Iteration zweimal mutieren werden darf...ich entscheide mich einfach mal für ja
					int selectedNeuronenindex = distNeuronIndex(generator);

					int sumNeurons = 0;
					for(int n=0;n<currentPopulation[indexFirstParent]->LayersCount;n++)//Iteration durch alle Layer
					{
						sumNeurons+=currentPopulation[indexFirstParent]->Layers[n].NeuronCount;//summiere Anzahl Neuronen bis zum momentanten Layer auf

						if(selectedNeuronenindex < sumNeurons)
						{
							NeuronIndex = selectedNeuronenindex - (sumNeurons -currentPopulation[indexFirstParent]->Layers[n].NeuronCount);

							for(int j=0;j<currentPopulation[indexFirstParent]->Layers[n].InputValuesCountWithBias;++j)//für jedes eingehende gewicht incl. Bias
							{
								currentPopulation[indexChild]->Layers[n].Neurons[NeuronIndex].Weights[j] += currentPopulation[indexChild]->GenerateRandomWeight(currentPopulation[indexChild]->Layers[n].InputValuesCountWithBias);
							}
							break; //nächstes Neuron
						}
					}
				}
			}

			//mse berechnen
			currentPopulation[indexChild]->CalculateMSE(dataContainer);
			//und an richtige stelle plazieren
			sortSingleElement(indexChild);

			MeasureResult[loopIndex].ExecuteTime = GetTimeMs64() - time64;
			MeasureResult[loopIndex].MeanSquareError = currentPopulation[0]->MeanSquareError;//speichern nur den besten

			//Abbruch Kriterium prüfen
			if(MeasureResult[loopIndex].MeanSquareError <= errorThreshold)
			{
				break;
			}

			//rouletteSum = 0;
		}
		MeasureFilledResultLenght = loopIndex;
	}
}