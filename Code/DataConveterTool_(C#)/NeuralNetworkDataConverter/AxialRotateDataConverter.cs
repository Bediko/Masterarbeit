using System;
using System.Collections.Generic;
using System.Globalization;
using System.Threading;

namespace NeuralNetworkDataConverter
{
    public class AxialRotateDataConverter
    {
        private const double MINVALUE = 0.1;
        private const double MAXVALUE = 0.9;

        private void SaveAll(List<string[]> values, string name)
        {
            System.IO.StreamWriter filetraining = new System.IO.StreamWriter(name, true);
            filetraining.Write(values.Count);
            filetraining.Write(" ");
            filetraining.Write(4);
            filetraining.Write(" ");
            filetraining.Write(1);
            filetraining.Write("\n");

            for (int j = 0; j < values.Count; j++)
            {
                filetraining.Write(values[j][0] + " ");
                filetraining.Write(values[j][1] + " ");
                filetraining.Write(values[j][2] + " ");
                filetraining.Write(values[j][3] + "\n");

                filetraining.Write(values[j][4] + "\n");
            }

            filetraining.Close();
        } 

        public void ConvertAxialRotatation()
        {
            Thread.CurrentThread.CurrentCulture = CultureInfo.CreateSpecificCulture("en-GB");
            string basePath = @"E:\Hochschule Niederrhein\Semester6\NN\data\axial-rotational";
            string line;

            List<string[]> values = new List<string[]>();

            // Read the file and display it line by line.
            System.IO.StreamReader filei = new System.IO.StreamReader(basePath + @"\feature_aor_all_features_p_0.5.csv");
            //int counter = 0;
            while ((line = filei.ReadLine()) != null)
            {
                string[] splitresult = line.Split(new char[] { ',' });

                if (splitresult[0] == "a")
                    splitresult[0] = "1";//AXIS
                else
                    splitresult[0] = "0";//ROTATION

                string tmp = splitresult[0];
                splitresult[0] = splitresult[1];
                splitresult[1] = splitresult[2];
                splitresult[2] = splitresult[3];
                splitresult[3] = splitresult[4];
                splitresult[4] = tmp;

                //counter++;
                //if (counter % 5 == 0 && splitresult[4] == "1")
                //    values.Add(splitresult);
                //else if (splitresult[4] == "0")
                //values.Add(splitresult);
                //if (splitresult[4] == "0")
                //{
                //    for (int k = 0; k < 8; k++)
                //    {
                //        string[] arr = new string[5];
                //        splitresult.CopyTo(arr, 0);
                //        values.Add(arr);
                //    }
                //}
                //else
                //    continue;
                Helper.Shuffle<string[]>(values);
            }

            //for (int i = 0; i < 5; ++i)
            //{
            //    List<string[]> train = new List<string[]>();
            //    List<string[]> test = new List<string[]>();
            //    int partCount = values.Count / 5;
            //    int testCount = values.Count - (partCount * (5 - 1)); // den rest als test daten
            //    for (int j = 0; j < values.Count; j++)
            //    {
            //        if (j >= i * partCount && j < (i * partCount) + testCount)
            //        {
            //            test.Add(values[j]);
            //            continue;
            //        }
            //        if (values[j][4] == "0")
            //        {
            //            for (int k = 0; k < 8; k++)
            //            {
            //                string[] arr = new string[5];
            //                values[j].CopyTo(arr, 0);
            //                train.Add(arr);
            //            }
            //        }
            //        train.Add(values[j]);
            //    }
            //    Helper.Shuffle<string[]>(train);
            //    SaveAll(train, "AR_01_to_09_Train"+i);
            //    SaveAll(test, "AR_01_to_09_Test"+i);
            //}
            //return;

            SaveAll(values, "AR_0_to_1");
            List<string[]> valuesScaled = new List<string[]>(values);
            for (int j = 0; j < values.Count; j++)
            {
                for (int i = 0; i < 5; i++)
                {
                    double val = Double.Parse(valuesScaled[j][i]);
                    double val2 = Helper.Scale(val, 0, 1, MINVALUE, MAXVALUE);
                    valuesScaled[j][i] = Math.Round(val2, 4, MidpointRounding.AwayFromZero).ToString();
                }
            }
            SaveAll(valuesScaled, "AR_01_09");
            filei.Close();
        }
    }
}