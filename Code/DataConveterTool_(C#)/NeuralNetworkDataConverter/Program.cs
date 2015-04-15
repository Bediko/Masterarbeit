using NeuralNetworkDataConverter;

namespace ReadMNIST
{
    /// <summary>
    /// Kleines Tool zum konvertieren der MNIST Daten in das Format das FANN und eigene NNTLib verwendet siehe z.B.: XOR.dat beispiel dafür
    /// </summary>
    public class Program
    {
        private static void Main(string[] args)
        {
            MNISTDataConverter conmnist = new MNISTDataConverter();
            conmnist.ConvertMNIST();
            AxialRotateDataConverter con = new AxialRotateDataConverter();
            con.ConvertAxialRotatation();
        } // Main
    } // Program
}