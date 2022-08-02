package fr.i360matt.ia;

import java.util.Arrays;

public class NNCache {

    private final NeuralNetwork nn;
    private final double[] hidden;
    private final double[] output;

    public NNCache (NeuralNetwork nn) {
        this.nn = nn;
        this.hidden = new double[this.nn.hidden_size];
        this.output = new double[this.nn.output_size];
    }

    public double[] forward (double[] input) {
        for (int i = 0; i < this.nn.output_size; i++) {
            double sum = 0;
            for (int j = 0; j < this.nn.hidden_size; j++) {

                double sum2 = 0;
                for (int o = 0; o < this.nn.input_size; o++) {
                    sum2 += input[o] * this.nn.weights_input_hidden[o][j];
                }
                sum2 += this.nn.bias_input_hidden[j];
                sum2 = sigmoid(sum2);
                hidden[j] = sum2;

                sum += sum2 * this.nn.weights_hidden_output[j][i];
            }
            sum += this.nn.bias_hidden_output[i];
            output[i] = sigmoid(sum);
        }
        return output;
    }

    public void train (double[] X, double[] Y) {
        forward(X);
        for (int k = 0; k < this.nn.hidden_size; k++) {
            double hiddenCache = hidden[k];
            double[] whoRow = this.nn.weights_hidden_output[k];
            double sum = 0;
            for (int l = 0; l < this.nn.output_size; l++) {
                double error_output = Y[l] - output[l];
                sum += error_output * whoRow[l];

                whoRow[l] += hiddenCache * error_output;
                this.nn.bias_hidden_output[l] += error_output;
            }

            double error_hidden = dsigmoid(hiddenCache) * sum;
            for (int l = 0; l < this.nn.input_size; l++) {
                this.nn.weights_input_hidden[l][k] += X[l] * error_hidden;
            }
            this.nn.bias_input_hidden[k] += error_hidden;
        }
    }


    public void train (double[][] X, double[][] Y, int iterations) {
        for (int i = 0; i < iterations; i++) {
            for (int j = 0; j < X.length; j++) {
                train(X[j], Y[j]);
            }
        }
    }

    public void fit (double[][] X, double[][] Y, int iterations) {
        for (int i = 0; i < iterations; i++) {
            // int index = (int) (Math.random() * X.length);
            int index = i % X.length;
            train(X[index], Y[index]);
        }
    }

    public double[] predict (double[] input) {
        return forward(input);
    }

    public static double exp (double x) {
        x = 1d + x / 256d;
        x *= x; x *= x; x *= x; x *= x;
        x *= x; x *= x; x *= x; x *= x;
        return x;
    }

    public static double sigmoid (double x) {
        return 1 / (1 + exp(-x));
    }

    public static double dsigmoid (double x) {
        return x * (1 - x);
    }

}
