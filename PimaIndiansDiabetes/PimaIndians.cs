﻿using System;
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
         * The purpose of the class is to predict diabetes among the pima indians. The class
         * can import data from file and manipulate the data, as well as creating a predictor 
         * and train it on the data
         */
        private List<double[]> dataset;
        private List<double[]> trainingset;
        private List<double[]> validationset;

        private static Random random = new Random();

        private static int NUMBER_OF_INPUTS = 8;
        private static int NUMBER_OF_OUTPUTS = 1;
        private static int NEURONS_IN_LAYER = 8;

        /*
         * CONSTRUCTORS
         */
        public PimaIndians() { }
        public PimaIndians(String path) {
            LoadData(path);
        }
        public PimaIndians(String path, double fraction) {
            LoadData(path);
            NormalizeDataset();
            DivideSet(fraction);
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
        public void DivideSet(double fraction) { 
            /*
             * Divide dataset into a training set and a validation set
             * fraction - the amount of data that will constitute the training set
             */
            this.trainingset = new List<double[]>();
            this.validationset = new List<double[]>();

            for (int i = 0; i < this.dataset.Count; i++) {
                if (i >= (this.dataset.Count * fraction))
                {
                    this.validationset.Add(this.dataset.ElementAt(i));
                }
                else
                {
                    this.trainingset.Add(this.dataset.ElementAt(i));
                }
            }
        }
        private List<double[]> permuteData(char fromSet)  { 
            /*
             * Permutation of data
             * fromChar - 't' if training set is to be permuted, 'v' if the 
             * validation set is to be permuted
             * Return a list of the permuted dataset
             */
            List<double[]> data = new List<double[]>();
            switch (fromSet) { 
                case 't':
                    //Create an array with all indices:
                    int[] idxT = createIndexarray(this.trainingset.Count);
                    NetworkUtils.Permute(ref idxT);
                    data = perm(this.trainingset, idxT);
                    break;
                case 'v':
                    //Create an array with all indices:
                    int[] idxV = createIndexarray(this.validationset.Count);
                    NetworkUtils.Permute(ref idxV);
                    data = perm(this.validationset, idxV);
                    break;
            }
            return data;
        }

        private int[] createIndexarray(int length) { 
            int[] idx = new int[length];
            for (int i = 0; i < length; i++) {
                idx[i] = i;
            }
            return idx;
        }
        private List<double[]> perm(List<double[]> list, int[] idx) {
            List<double[]> permList = new List<double[]>();
            double nDatapoints = NUMBER_OF_OUTPUTS + NUMBER_OF_INPUTS;
            for (int i = 0; i < idx.Length; i++){
                double[] x = list.ElementAt(idx[i]);
                permList.Add(x);
            }
            return permList;
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
        public void PredictDiabetes() {
            /*
             * Create and train a neural network to predict if an individual has diabetes 
             * among the pima indians.
             * The method print out the prediction error to the console for the training and 
             * validation set to demonstrate how the network improve
             */
            FeedForwardNeuralNetwork network = new FeedForwardNeuralNetwork(NUMBER_OF_INPUTS, NUMBER_OF_OUTPUTS, NEURONS_IN_LAYER);
            for (int i = 0; i < 80000; i++)
            {
                //Train network on trainingset 
                network.Train(permuteData('t'));
                
                //Print error of trainingset
                network.PrintAccuracy(permuteData('t')); //Root-mean-square-error
                //Measure error of validationset
            }
        } //improve method! for example: visualize the learning, quit training before overfitting etc.
    }
}
