#ifndef DBNLayer_H
#define DBNLayer_H

#include "DBNNeuron.h"
#include "Layer.h"
namespace NNTLib
{
	class DBNLayer : public Layer
	{
	private:
		void copy(const DBNLayer &that);
	public:
		
		DBNNeuron *Neurons;
		void Init(int inputsize, int neuronCount);
		void Forwardweightsinit(int inputsize, DBNLayer* Layerup);
		DBNLayer();
		~DBNLayer();
		DBNLayer(const DBNLayer &that);
		DBNLayer& operator= (const DBNLayer &that);
	};
}
#endif