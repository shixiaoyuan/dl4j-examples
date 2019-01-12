package mycode.stock;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class StockPredict {
    private static final int IN_NUM = 6;
    private static final int OUT_NUM = 1;
    private static final int Epochs = 100;

    private static final int lstmLayerSize = 15;


    public static MultiLayerNetwork getNetModel(int nIn, int nOut) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(new RmsProp(0.1))
            .seed(12345)
            .l2(0.001)
            .weightInit(WeightInit.XAVIER)
            .list()
            .layer(0, new LSTM.Builder().nIn(nIn).nOut(lstmLayerSize)
                .activation(Activation.TANH).build())
            .layer(1, new LSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
                .activation(Activation.TANH).build())
            .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY)
                .nIn(lstmLayerSize).nOut(nOut).build())
            .pretrain(false).backprop(true)
            .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));
        return net;
    }

    public static void train(MultiLayerNetwork net, StockDataIterator iterator){
        //迭代训练
        for(int i=1;i<=20;i++) {
            DataSet dataSet = null;
            while (iterator.hasNext()) {
                dataSet = iterator.next();
                net.fit(dataSet);
            }
            iterator.reset();
            System.out.println();
            System.out.println("=================>完成第"+i+"次完整训练");
            INDArray initArray = getInitArray(iterator);

            System.out.println("预测结果：");
            for(int j=0;j<100;j++) {
                INDArray output = net.rnnTimeStep(initArray);
                System.out.println(output.shape());
//              for(int seq=0;seq<30;++seq) {
//                    System.out.print(output.getDouble(seq)*iterator.getMaxArr()[1]+" ");
//                }

                System.out.print(output.getDouble(0)*iterator.getMaxArr()[1]+" ");

            }
            net.rnnClearPreviousState();
        }
    }

    private static INDArray getInitArray(StockDataIterator iter){
        double[] maxNums = iter.getMaxArr();
        INDArray initArray = Nd4j.zeros(1, 6, 1);
        initArray.putScalar(new int[]{0,0,0}, 3433.85/maxNums[0]);
        initArray.putScalar(new int[]{0,1,0}, 3445.41/maxNums[1]);
        initArray.putScalar(new int[]{0,2,0}, 3327.81/maxNums[2]);
        initArray.putScalar(new int[]{0,3,0}, 3470.37/maxNums[3]);
        initArray.putScalar(new int[]{0,4,0}, 304197903.0/maxNums[4]);
        initArray.putScalar(new int[]{0,5,0}, 3.8750365e+11/maxNums[5]);
        return initArray;
    }
    public static void main(String[] args) {
        String inputFile = "/Users/shiyuan/Desktop/download/sh000001.csv";
        int batchSize = 1;
        int exampleLength = 30;
        //初始化深度神经网络
        StockDataIterator iterator = new StockDataIterator();
        iterator.loadData(inputFile,batchSize,exampleLength);

        MultiLayerNetwork net = getNetModel(IN_NUM,OUT_NUM);
        train(net, iterator);
    }
}
