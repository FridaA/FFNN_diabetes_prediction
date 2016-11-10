using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace PimaIndiansDiabetes
{
    public static class NetworkUtils
    {
        /*
         * Feed forward neural network utilities such as useful mathematical formulas 
         */
        private static double MIN_INITIAL_WEIGHT_INTERVAL = -1;
        private static double MAX_INITIAL_WEIGHT_INTERVAL = 1;
        private static Random random = new Random();

        /*
         * METHODS
         */
        private static double randomInInterval(double minValue, double maxValue)
        {
            /*
             * Generate a random number in the interval [minValue, maxValue[
             * minValue - minimum value in interval
             * maxValue - maximum value in interval
             * Return a random number
             */
            double value = random.NextDouble();
            return minValue + (value * (maxValue - minValue));
        }
        public static double[,] InitRandomArray(int length, int width)
        {
            /*
             * Creates an matrix and fills it with random numbers in the interval 
             * [MIN_INITIAL_WEIGHT_INTERVAL, MAX_INITIAL_WEIGHT_INTERVAL[
             * length - the length of the matrix
             * width - the width of the matrix
             * Returns a matrix with randomized values in it
             */
            double[,] matrix = new double[length, width];
            for (int i = 0; i < length; i++)
            {
                for (int n = 0; n < width; n++)
                {
                    matrix[i, n] = randomInInterval(MIN_INITIAL_WEIGHT_INTERVAL, MAX_INITIAL_WEIGHT_INTERVAL);
                }
            }
            return matrix;
        }
        public static double[] InitRandomArray(int length) {
            double[] array = new double[length];
            for (int i = 0; i < length; i++) {
                array[i] = randomInInterval(MIN_INITIAL_WEIGHT_INTERVAL, MAX_INITIAL_WEIGHT_INTERVAL);
            }
            return array;
        }
        public static double ActivationFunction(double value, double constant = 1) {
            /* 
             * Calculates and returns the activation function (the sigmoid)
             */
            return Math.Tanh(constant * value);
                //1 / (1 + Math.Exp(-constant * value));
        }
        public static double ActivationFunctionDerivative(double value, double constant = 1) { 
            /* 
             * Calculates and returns the derivative of the activation function
             */
            double numerator = 2*Math.Cosh( constant * value );
            double denominator = Math.Cosh(constant * 2 * value) + 1;
            return Math.Pow(numerator/denominator, 2);
                 //constant*Math.Exp(constant * value)/Math.Pow(Math.Exp(constant * value) + 1,2);
        }
        public static double[,] TransposeMatrix(double[,] matrix) {
            double[,] transposedMatrix = new double[matrix.GetLength(1), matrix.GetLength(0)];
            for (int n = 0; n < matrix.GetLength(1); n++) {
                for (int m = 0; m < matrix.GetLength(0); m++ )
                {
                    transposedMatrix[n, m] = matrix[m, n];
                }
            }
            return transposedMatrix;
        }
        public static double RootMeanSquare(double[] errors) {
            double sum = 0;
            for (int i = 0; i < errors.Length; i++) {
                sum = sum + 2 * errors[i];
            }
            double rms = Math.Sqrt(sum / errors.Length);
            return rms;
        }
        private static double sum(double[] arr) { 
            /*
             * Sum all elements in a double array
             * arr - the array from which the element is summed up
             */
            double sum = 0;
            for(int i = 0; i < arr.Length; i++) {
                sum = sum + arr[i];
            }
            return sum;
        }
        private static double standardDeviation(double[] arr, double mean) {
            /*
             * Gives the standard deviation of array arr
             */
            double sum = 0;
            for (int i = 0; i < arr.Length; i++) {
                sum = sum + Math.Pow( Math.Abs(arr[i] - mean), 2); 
            }
            return Math.Sqrt( sum/ arr.Length );
        }
        public static double[] Normalize(double[] arr)
        {
            /*
             * Gives the normalized array of arr
             */
            double meanValue = sum(arr) / arr.Length;
            double std = standardDeviation(arr, meanValue);

            double[] normalizedArr = new double[arr.Length];
            for (int i = 0; i < arr.Length; i++)
            {
                normalizedArr[i] = (arr[i] - meanValue) / std;
            }
            return normalizedArr;
        }
        public static int Sign(double value) { 
            /*
             * Returns 1 if: value is greater than 0
             * Returns 0 if: value equals 0
             * Returns -1 if: value is less than 0
             */           
            if (value > 0)
                 return 1;
            else if (value == 0)
                return 0;
            else
                return -1;            
        }
        public static void Permute(ref int[] arr) { 
            int listLength = arr.Length -1;
            //permute(arr, 0, listLength);
            for (int i = listLength; i > 0; i--) {
                int j = random.Next(i);
                swap(ref arr[j], ref arr[i]);
            }
        }
        private static void swap(ref int a, ref int b) {
            if (a.Equals(b))
                return;
            int tmp = a;
            a = b;
            b = tmp;
        }
        private static void permute(int[] arr, int recursionDepth, int maxDepth) {
            //for (int i = 0; i < )
        }
    }
}
