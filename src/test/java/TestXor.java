import fr.i360matt.ia.NNCache;
import fr.i360matt.ia.NeuralNetwork;

import java.io.IOException;
import java.util.Arrays;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;


public class TestXor {

    public static void main(String[] args) throws InterruptedException, IOException {

        long start = System.currentTimeMillis();
        run();
        long end = System.currentTimeMillis();

        System.out.println("Time: " + (end - start) + "ms");

    }


    public static void run () {

        double[][] input = new double[][] {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1},
            {0.5, 0.5},
            {0.3, 0.3}
        };

        double[][] output = new double[][] {
            {0},
            {1},
            {1},
            {0},
            {0.5},
            {0.9}
        };

        int threads = 2;
        int iters = 1_000_000;

        NeuralNetwork network = new NeuralNetwork(2, 5, 1);


        ForkJoinPool commonPool = new ForkJoinPool(4);
        ForkJoinTask<?>[] futures = new ForkJoinTask[threads];
        for (int i = 0; i < threads; i++) {
            futures[i] = commonPool.submit(() ->  {
                NNCache cache = new NNCache(network);
                cache.fit(input, output, iters);
            });
        }

        for (ForkJoinTask<?> future : futures) {
            future.join();
        }




        for (int i = 0; i < input.length; i++) {
            NNCache cache = new NNCache(network);
            double[] output_ = cache.predict(input[i]);
            System.out.println(Arrays.toString(input[i]) + " -> " + Arrays.toString(output_));
        }

    }

}
