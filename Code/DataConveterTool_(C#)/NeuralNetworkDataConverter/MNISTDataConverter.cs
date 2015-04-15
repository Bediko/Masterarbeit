using System;
using System.Globalization;
using System.IO;
using System.Threading;

namespace NeuralNetworkDataConverter
{
    public class MNISTDataConverter
    {
        private const double MINVALUE = 0.1;
        private const double MAXVALUE = 0.9;

        public void ConvertMNIST()
        {
            Thread.CurrentThread.CurrentCulture = CultureInfo.CreateSpecificCulture("en-GB");
            double RANGE = Math.Abs(MAXVALUE) + Math.Abs(MINVALUE);

            try
            {
                const char LF = (char)10;//LF statt CRLF bei newline
                Console.WriteLine("\nBegin\n");
                string basePath = @"E:\Hochschule Niederrhein\Semester6\NN\data\mnist";
                //t10k- or train-
                FileStream ifsLabels = new FileStream(basePath + @"\train-labels.idx1-ubyte", FileMode.Open); // test labels
                FileStream ifsImages = new FileStream(basePath + @"\train-images.idx3-ubyte", FileMode.Open); // test images
                BinaryReader brLabels = new BinaryReader(ifsLabels);
                BinaryReader brImages = new BinaryReader(ifsImages);

                /*
                [offset] [type]          [value]          [description]
                0000     32 bit integer  0x00000803(2051) magic number
                0004     32 bit integer  10000            number of images
                0008     32 bit integer  28               number of rows
                0012     32 bit integer  28               number of columns
                0016     unsigned byte   ??               pixel
                0017     unsigned byte   ??               pixel
                ........
                xxxx     unsigned byte   ??               pixel
                 */
                //erstmal die ganzen magic sachen und meta daten skippen
                int magic1 = Helper.ReadBigInt32(brImages); //magic number
                int numImages = Helper.ReadBigInt32(brImages);//number of images
                int numRows = Helper.ReadBigInt32(brImages);//number of rows
                int numCols = Helper.ReadBigInt32(brImages);//number of columns
                /*
                [offset] [type]          [value]          [description]
                0000     32 bit integer  0x00000801(2049) magic number (MSB first)
                0004     32 bit integer  10000            number of items
                0008     unsigned byte   ??               label
                0009     unsigned byte   ??               label
                ........
                xxxx     unsigned byte   ??               label
                 */
                /*
                    Result Datei sollte etwa so aussehen:
                    4 2 1
                    -1 -1
                    -1
                    -1 1
                    1
                    1 -1
                    1
                    1 1
                    -1

                    The first line consists of three numbers: The first is the number of training pairs in the file, the second is the number of
                    inputs and the third is the number of outputs.  The rest of the file is the actual training data, consisting of one line with
                    inputs, one with outputs etc.
                 */
                //wieder erstmal magic numbers usw überspringen
                int magic2 = Helper.ReadBigInt32(brLabels);
                int numLabels = Helper.ReadBigInt32(brLabels);

                if (numImages != numLabels)
                {
                    Console.WriteLine("Error: Anzahl images und labes unterscheiden sich");
                    Console.ReadKey();
                    return;
                }

                string filename = @"MNIST_TRAIN_" + MINVALUE.ToString() + "_" + MAXVALUE.ToString();

                if (File.Exists(filename))
                    File.Delete(filename);

                using (System.IO.StreamWriter file = new System.IO.StreamWriter(filename, true))
                {
                    file.Write(numImages);
                    file.Write(" ");
                    file.Write(numRows * numCols);
                    file.Write(" ");
                    file.Write(GetResultNeuronString(0).Length);
                    file.Write(LF);

                    for (int di = 0; di < numImages; ++di)
                    {
                        for (int i = 0; i < numRows; ++i)
                        {
                            for (int j = 0; j < numCols; ++j)
                            {
                                byte b = brImages.ReadByte();
                                double val = 0;

                                val = Helper.Scale(b, 0, 255, MINVALUE, MAXVALUE);

                                file.Write(val);
                                if (i * j < (numRows - 1) * (numCols - 1))
                                    file.Write(" ");
                            }
                        }

                        byte lbl = brLabels.ReadByte();
                        file.Write(LF);
                        var res = GetResultNeuronString(lbl);
                        for (int i = 0; i < res.Length; i++)
                        {
                            file.Write(res[i]);
                            if (i != res.Length - 1)
                                file.Write(" ");
                        }
                        file.Write(LF);
                    }
                }

                ifsImages.Close();
                brImages.Close();
                ifsLabels.Close();
                brLabels.Close();

                Console.WriteLine("\nEnde\n");
                Console.ReadKey();
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                Console.ReadKey();
            }
        }

        private double[] GetResultNeuronString(int Value)
        {
            double[] r = { MINVALUE, MINVALUE, MINVALUE, MINVALUE, MINVALUE, MINVALUE, MINVALUE, MINVALUE, MINVALUE, MINVALUE };
            r[Value] = MAXVALUE;
            return r;
        }
    }
}