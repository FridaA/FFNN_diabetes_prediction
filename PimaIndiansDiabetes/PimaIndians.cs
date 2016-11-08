using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using System.Globalization;
using System.IO;

namespace PimaIndiansDiabetes
{
    public class PimaIndians
    {
        /*
         * 
         */
        private List<double[]> dataset;
        private List<double[]> trainingset;
        private List<double[]> validationset;

        private static Random random = new Random();

        private static int NUMBER_OF_INPUTS = 8;
        private static int NUMBER_OF_OUTPUTS = 1;
        private static int NEURONS_IN_LAYER = 5;

        /*
         * CONSTRUCTORS
         */
        public PimaIndians() { }
        public PimaIndians(String path) {
            LoadData(path);
        }
        public PimaIndians(String path, double percent) {
            LoadData(path);
            NormalizeDataset();
            DivideSet(percent);
        }
        /*
         * METHODS
         */
        public void LoadData(String path) {
            /*
             *  Load data from file
             *  path - the file location
             */
            StreamReader sr = new StreamReader(path);
            this.dataset = new List<double[]>();

            string line;
            char[] splitChars = { ',' };
            while ((line = sr.ReadLine()) != null) {
                string[] parameters = line.Split(splitChars);
                double[] data = new double[parameters.Length];
                for (int i = 0; i < data.Length; i++) {
                    data[i] = double.Parse(parameters[i], CultureInfo.InvariantCulture.NumberFormat);
                }
                this.dataset.Add(data);
            }
        }
        public void DivideSet(double percent) { 
            /*
             * Divide dataset into a training set and a validation set
             * percent - the amount of data that will constitute the training set
             */
            this.trainingset = new List<double[]>();
            this.validationset = new List<double[]>();

            for (int i = 0; i < this.dataset.Count; i++) {
                if (i >= (this.dataset.Count * percent))
                {
                    this.validationset.Add(this.dataset.ElementAt(i));
                }
                else
                {
                    this.trainingset.Add(this.dataset.ElementAt(i));
                }
            }
        }
        private List<double[]> permuteData(char fromSet) { //NOT FINISHED!
            /*
             * Permutation of data
             * fromChar - 't' if training set is to be permuted, 'v' if the 
             * validation set is to be permuted
             * Return a list of the permuted dataset
             */
            List<double[]> data = new List<double[]>();
            return data;
        }
        private List<double[]> getStochasticData(int numberOfDatapoints, char fromSet)
        {
            /*
             * Get a set of stochastic datapoints from one of the data sets
             * numberOfDatapoints - the number of datapoints to get from the set
             * fromSet - determines which set to take datapoints from: 't' for training set, 
             * 'v' for validationset
             * Return a list of datapoints with length numberOfDatapoints
             */
            List<double[]> data = new List<double[]>();
            switch (fromSet)
            {
                case 't':
                    for (int i = 0; i < numberOfDatapoints; i++)
                    {
                        data.Add(this.trainingset.ElementAt(random.Next(this.trainingset.Count)));
                    }
                    break;
                case 'v':
                    for (int i = 0; i < numberOfDatapoints; i++)
                    {
                        data.Add(this.validationset.ElementAt(random.Next(this.validationset.Count)));
                    }
                    break;
            }
            return data;
        }
        public void NormalizeDataset() {             
            double[] inputs;
            double[] desiredOutputs;
            double[] normalizedInputs;
            foreach (double[] iopair in this.dataset) {
                inputs = iopair.Take(NUMBER_OF_INPUTS).ToArray();
                desiredOutputs = iopair.Skip(NUMBER_OF_INPUTS).Take(NUMBER_OF_OUTPUTS).ToArray();
                
                //Normalize input data
                normalizedInputs = NetworkUtils.Normalize(inputs);

                //Update dataset with the normalized data
                for(int i = 0; i < NUMBER_OF_INPUTS; i++) 
                    iopair[i] = normalizedInputs[i];
            }
        }
        public void PredictDiabetes() { //NOT FINISHED!
            /*
             * Create and train a neural network to predict if an individual has diabetes 
             * among the pima indians.
             * The method print out the prediction error to the console for the training and 
             * validation set to demonstrate how the network improve
             */
            FeedForwardNeuralNetwork network = new FeedForwardNeuralNetwork(NUMBER_OF_INPUTS, NUMBER_OF_OUTPUTS, NEURONS_IN_LAYER);
            for (int i = 0; i < 8000; i++)
            {
                network.Train(getStochasticData(this.trainingset.Count, 't'));
                network.PrintAccuracy(getStochasticData(this.trainingset.Count, 't'));
            }
        }
    }
}
