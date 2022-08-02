package fr.i360matt.ia;


import java.util.Random;

public class NeuralNetwork {

    protected final Random random;

    protected final int input_size;
    protected final int hidden_size;
    protected final int output_size;

    protected final double[][] weights_input_hidden;
    protected final double[][] weights_hidden_output;

    protected final double[] bias_input_hidden;
    protected final double[] bias_hidden_output;

    public NeuralNetwork (int input_size, int hidden_size, int output_size) {
        this.random = new Random();

        this.input_size = input_size;
        this.hidden_size = hidden_size;
        this.output_size = output_size;

        this.weights_input_hidden = new double[input_size][hidden_size];
        this.weights_hidden_output = new double[hidden_size][output_size];

        this.bias_input_hidden = new double[hidden_size];
        this.bias_hidden_output = new double[output_size];

        for (int i = 0; i < input_size; i++) {
            double[] row = this.weights_input_hidden[i];
            for (int j = 0; j < hidden_size; j++) {
                row[j] = random.nextDouble() * 2 - 1;
            }
        }

        for (int i = 0; i < hidden_size; i++) {
            double[] row = this.weights_hidden_output[i];
            for (int j = 0; j < output_size; j++) {
                row[j] = random.nextDouble() * 2 - 1;
            }
        }

        for (int i = 0; i < hidden_size; i++) {
            this.bias_input_hidden[i] = random.nextDouble() * 2 - 1;
        }

        for (int i = 0; i < output_size; i++) {
            this.bias_hidden_output[i] = random.nextDouble() * 2 - 1;
        }

    }


}
