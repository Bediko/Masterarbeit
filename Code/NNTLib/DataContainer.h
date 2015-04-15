#ifndef DataContainer_H
#define DataContainer_H

#include <sstream>
#include <fstream>
#include <stdexcept>

namespace NNTLib
{
	class DataContainer
	{
	private:
		void init();
		void init(int dataCount,int inputCount,int outputCount);
		void freeMem();
		void copy(const DataContainer &that);
	public:
		/// <summary>
		/// The data count
		/// </summary>
		int DataCount;
		/// <summary>
		/// The input count
		/// </summary>
		int InputCount;
		/// <summary>
		/// The output count
		/// </summary>
		int OutputCount;

		/// <summary>
		/// The data input
		/// </summary>
		double **DataInput;
		/// <summary>
		/// The data output
		/// </summary>
		double **DataOutput;

		void CopyData(const DataContainer &src,int startindexDst,int startindexSource,int lenght);
		DataContainer();
		void LoadFile(const char* file);
		void Init(int dataCount,int inputCount,int outputCount);
		~ DataContainer();
		DataContainer(const DataContainer &that);
		DataContainer& operator= (const DataContainer &that);
	};
}
#endif
