using System;
using System.Collections.Generic;
using System.IO;

namespace NeuralNetworkDataConverter
{
    public class Helper
    {

        /// <summary>
        /// Scales the specified element to scale.
        /// </summary>
        /// <param name="elementToScale">The element to scale.</param>
        /// <param name="rangeMin">The range minimum.</param>
        /// <param name="rangeMax">The range maximum.</param>
        /// <param name="scaledRangeMin">The scaled range minimum.</param>
        /// <param name="scaledRangeMax">The scaled range maximum.</param>
        /// <returns></returns>
        public static double Scale(double elementToScale, double rangeMin, double rangeMax, double scaledRangeMin, double scaledRangeMax)
        {
            var scaled = scaledRangeMin + ((elementToScale - rangeMin) * (scaledRangeMax - scaledRangeMin) / (rangeMax - rangeMin));
            return scaled;
        }

        /// <summary>
        /// Shuffles the specified list.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="list">The list.</param>
        public static void Shuffle<T>(IList<T> list)
        {
            Random rng = new Random();
            int n = list.Count;
            while (n > 1)
            {
                n--;
                int k = rng.Next(n + 1);
                T value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
        }

        /// <summary>
        /// Reads the big int32.
        /// </summary>
        /// <param name="br">The br.</param>
        /// <returns></returns>
        public static int ReadBigInt32(BinaryReader br)
        {
            var bytes = br.ReadBytes(sizeof(Int32));
            if (BitConverter.IsLittleEndian) Array.Reverse(bytes);
            return BitConverter.ToInt32(bytes, 0);
        }
    }
}