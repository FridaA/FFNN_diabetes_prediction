using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace PimaIndiansDiabetes
{
    public class FeedForwardNeuralNetwork
    {
        /*
         * Feed forward neural network with two hidden layers and one output (version 1)
         * Neurons are McCulloch-Pitts neurons
         */
        private int numberOfInputs; //Number of inputs to network
        private int numberOfOutputs; //Number of outputs from network
        private int numberOfHiddenLayers = 2; //Number of hidden neuron layers in network
        private int neuronsInLayer; //the constant number of neurons in each hidden layer
        
        private double[,] ihWeights; //weights between input and first hidden layer
        private double[,] hhWeights; //weights between first and second hidden layer 
        private double[,] hoWeights; //weights between second hidden layer and output

        private double[] ihThresholds; //thresholds for the neurons in the first hidden layer
        private double[] hhThresholds; //thresholds for the neurons in the second hidden layer
        private double[] hoThresholds; //thresholds for the output neurons

        private static double STEP_LENGTH = 0.01;

        /*
         * CONSTRUCTORS
         */
        public FeedForwardNeuralNetwork(int inputs, int outputs, int neuronsInLayer) {
            /*
             * Constructor
             * inputs - the number of inputs to the network
             * outputs - the number of outputs from the network
             * neuronsInLayer - the number of neurons in the hidden layer
             */
            this.numberOfInputs = inputs;
            this.numberOfOutputs = outputs;
            this.neuronsInLayer = neuronsInLayer;
            initializeNetwork();
        }
        /*
         * METHODS
         */
        public void Train(List<double[]> dataset) { 
            /*
             * Train the network on the dataset by changing weights and thresholds 
             * to better estimate the outputs in the dataset
             * dataset - a set of input-output pairs 
             */
            double[] inputs;
            double[] desiredOutputs;
            double[] networkOutputs;
            double[] errorSignal;
            foreach (double[] iopair in dataset) {
                inputs = iopair.Take(this.numberOfInputs).ToArray();
                desiredOutputs = iopair.Skip(this.numberOfInputs).Take(this.numberOfOutputs).ToArray();
                //Feed forward:
                networkOutputs = computeNetworkOutputs(inputs);
                errorSignal = computeErrorSignal(desiredOutputs, networkOutputs);
                //Back propagation:
                updateWeights(inputs, errorSignal);
            }
        }
        private void initializeNetwork() { 
            /*
             * Set weights and thresholds to random numbers in the interval 
             * [MIN_INITIAL_WEIGHT_INTERVAL, MAX_INITIAL_WEIGHT_INTERVAL[
             */
            this.ihWeights = NetworkUtils.InitRandomArray(this.numberOfInputs, this.neuronsInLayer);
            this.hhWeights = NetworkUtils.InitRandomArray(this.neuronsInLayer, this.neuronsInLayer);
            this.hoWeights = NetworkUtils.InitRandomArray(this.neuronsInLayer, this.numberOfOutputs);
            this.ihThresholds = NetworkUtils.InitRandomArray(this.neuronsInLayer);
            this.hhThresholds = NetworkUtils.InitRandomArray(this.neuronsInLayer);
            this.hoThresholds = NetworkUtils.InitRandomArray(this.numberOfOutputs);
        }
        private double[] computeWeightedSum(double[] inputs, double[,] weights, double[] threshold) { 
            /*
             * Method returns the weighted sum into each neuron in a layer
             * inputs - the inputs to the layer
             * weights - the weights connecting inputs and neurons
             * threshold - the neuron threshold
             */
            double[] sum = new double[weights.GetLength(1)];
            for (int n = 0; n < weights.GetLength(1); n++) { //Step through every output neuron
                for (int m = 0; m < inputs.GetLength(0); m++) { //Step through every input neuron
                    sum[n] = sum[n] + weights[m, n] * inputs[m];
                }
                sum[n] = sum[n] - threshold[n];
            }
            return sum;
        }
        private double[] computeGradientSum(double[] gradient, double[,] weights) { 
            /*
             * Calculation of the gradient sum that appear in the backpropagation, in the calculations 
             * for weight change in hidden layers
             */
            double[] sum = new double[weights.GetLength(0)];
            for (int i = 0; i < gradient.Length; i++) {
                for (int n = 0; n < sum.Length; n++) {
                    sum[n] = sum[n] + weights[n, i] * gradient[i];
                }
            }
            return sum;
        }
        private double[] computeNeuronOutputs(double[] inputs, double[,] weights, double[] thresholds) {
            /*
             * Calculation of the output of every neuron in a layer 
             * inputs - the inputs to the layer
             * weights - the weights connecting inputs and neurons
             * Returns the computed output from a leyer in the network
             */
            double[] sum = computeWeightedSum(inputs, weights, thresholds);
            double[] output = new double[sum.Length];
            for (int i = 0; i < sum.Length; i++){
                output[i] = NetworkUtils.ActivationFunction(sum[i]);
            }
            return output;
        }
        private double[] computeNetworkOutputs(double[] inputs) { 
        /*
         * Calculates the network output given a specific input to the network (only two hidden layers 
         * in version 1)
         * inputs - the set of inputs in to the network
         * Returns the computed output from the network
         */
            double[,] weights = this.ihWeights;
            double[] thresholds = this.ihThresholds;
            double[] nextInput = inputs;
            for (int i = 0; i < this.numberOfHiddenLayers; i++)
            {
                double[] layerOutput = computeNeuronOutputs(nextInput, weights, thresholds);
                weights = this.hhWeights;
                thresholds = this.hhThresholds;
                nextInput = layerOutput;
            }
            weights = this.hoWeights;
            thresholds = this.hoThresholds;

            return computeNeuronOutputs(nextInput, weights, thresholds);
        }
        private double[] computeErrorSignal(double[] desired, double[] actual) { 
            /*
             * Calculate the error signal 
             * desired - the desired output from the network for a specific input
             * actual - the actual output from the network
             * Returns the error between the two for each output neuron
             */
            double[] error = new double[desired.Length];
            for (int i = 0; i < desired.Length; i++) {
                error[i] = desired[i] - actual[i]; //Math.Pow(desired[i] - actual[i], 2)/2;
            }
            return error;
        }
        private double computeTotalError(double[] desired, double[] actual) {
            /*
             * Calculate the total error 
             * desired - the desired output from the network for a specific input
             * actual - the actual output from the network
             * Returns the total error
             */
            double error = 0;
            for (int i = 0; i < desired.Length; i++)
            {
                error = error + Math.Pow(desired[i] - actual[i], 2);
                    //Math.Abs(desired[i] - NetworkUtils.Sign(actual[i])) / 2;
            }
            return error;
        }//not in use
        private double computeTotalError(double[] e) { 
            /*
             * Compute the total error of e
             */
            double error = 0;
            for (int i = 0; i < e.Length; i++) {
                error = error + Math.Pow(e[i], 2);
            }
            return error;
        }
        private void updateWeights(double[] inputs, double[] error) { 
            /*
             * Backpropagation to calculate and update all weights and thresholds
             * inputs - the network input
             * error - the prediction error for this input
             */

            // The weighted sums and output for the neurons in the first hidden layer
            double[] ihSum = computeWeightedSum(inputs, this.ihWeights, this.ihThresholds);
            double[] ihOut = computeNeuronOutputs(inputs, this.ihWeights, this.ihThresholds);

            // Weighted sums and output from the second hidden layer
            double[] hhSum = computeWeightedSum(ihOut, this.hhWeights, this.hhThresholds);
            double[] hhOut = computeNeuronOutputs(ihOut, this.hhWeights, this.hhThresholds);

            // The weighted sums and output of the network output neurons 
            double[] hoSum = computeWeightedSum(hhOut, this.hoWeights, this.hoThresholds);
            double[] hoOut = computeNeuronOutputs(hhOut, this.hoWeights, this.hoThresholds);

            //Update of weights between output and second hidden layer
            double[] hoGradient = new double[this.numberOfOutputs];
            for (int i = 0; i < this.numberOfOutputs; i++)
            { //For each network output
                //Calculate the local gradient
                hoGradient[i] = error[i] * NetworkUtils.ActivationFunctionDerivative(hoSum[i]);
                for (int n = 0; n < this.neuronsInLayer; n++)
                { //For each neuron in hidden layer
                    //Update weights with gradient descent to approach the desired output
                    this.hoWeights[n, i] = this.hoWeights[n, i] + STEP_LENGTH * hoGradient[i] * hhOut[n];
                }
                this.hoThresholds[i] = this.hoThresholds[i] - STEP_LENGTH * hoGradient[i];
            }
            //Update of weights between hidden layers
            double[] hhGradient = new double[this.neuronsInLayer];
            double[] hhGradientSum = computeGradientSum(hoGradient, this.hoWeights);
                //computeWeightedSum(hoGradient, NetworkUtils.TransposeMatrix(this.hoWeights), this.hhThresholds);
            for (int i = 0; i < this.neuronsInLayer; i++) { //for each neuron in second hidden layer 
                //Calculate the local gradient
                hhGradient[i] = NetworkUtils.ActivationFunctionDerivative(hhSum[i]) * hhGradientSum[i];
                for (int j = 0; j < this.neuronsInLayer; j++) { //for each neuron in first hidden layer
                    //Update weights and thresholds with gradient descent
                    this.hhWeights[j, i] = this.hhWeights[j, i] + STEP_LENGTH * hhGradient[i] * ihOut[j];
                    //this.hhThresholds[j] = this.hhThresholds[j] + STEP_LENGTH * hhGradient[i]; 
                }
                this.hhThresholds[i] = this.hhThresholds[i] - STEP_LENGTH * hhGradient[i];
            }

            //Update of weights between first hidden layer and input layer
            double[] ihGradient = new double[this.numberOfInputs];
            double[] ihGradientSum = computeGradientSum(hhGradient, this.hhWeights);
                //computeWeightedSum(hhGradient, NetworkUtils.TransposeMatrix(this.hhWeights), this.ihThresholds);
            for (int i = 0; i < this.neuronsInLayer; i++)
            { //For each neuron in hidden layer
                //Calculate the local gradient
                ihGradient[i] = NetworkUtils.ActivationFunctionDerivative(ihSum[i]) * ihGradientSum[i];
                for (int j = 0; j < this.numberOfInputs; j++)
                { //For each network input
                    //Update weights with gradient descent to approach the desired output
                    this.ihWeights[j, i] = this.ihWeights[j, i] + STEP_LENGTH * ihGradient[i] * inputs[j];
                    //this.ihThresholds[j] = this.ihThresholds[j] + STEP_LENGTH * ihGradient[i];
                }
                this.ihThresholds[i] = this.ihThresholds[i] - STEP_LENGTH * ihGradient[i];
            }
        } //improve implementation
        public void PrintAccuracy(List<double[]> dataset) { 
            /*
             * Print the accuracy of the network output prediction on the dataset
             * dataset - a set of input-output pairs 
             */
            List<double> totalErrors = new List<double>();
            double[] inputs;
            double[] desiredOutputs;
            double[] networkOutputs;
            foreach (double[] iopair in dataset) {
                inputs = iopair.Take(this.numberOfInputs).ToArray();
                desiredOutputs = iopair.Skip(this.numberOfInputs).Take(this.numberOfOutputs).ToArray();
                //Feed forward:
                networkOutputs = computeNetworkOutputs(inputs);
                totalErrors.Add(computeTotalError(desiredOutputs, networkOutputs));
            }
            double rootMeanSquareError = NetworkUtils.RootMeanSquare(totalErrors.ToArray());
            Console.WriteLine(rootMeanSquareError);
        } //not in use
        public void PrintTotalError(List<double[]> dataset) { 
            /*
             * Calculate the total error on the dataset and print it to the console
             */
            List<double> errors = new List<double>();
            double[] inputs;
            double[] desiredOutputs;
            double[] networkOutputs;
            foreach (double[] iopair in dataset)
            {
                inputs = iopair.Take(this.numberOfInputs).ToArray();
                desiredOutputs = iopair.Skip(this.numberOfInputs).Take(this.numberOfOutputs).ToArray();
                //Feed forward:
                networkOutputs = computeNetworkOutputs(inputs);
                double[] e = computeErrorSignal(desiredOutputs, networkOutputs);
                errors.Add(e[0]); //temporary solution
            }
            //double rootMeanSquareError = NetworkUtils.RootMeanSquare(totalErrors.ToArray());
            Console.WriteLine(computeTotalError(errors.ToArray()));
        }
       
    }
}
