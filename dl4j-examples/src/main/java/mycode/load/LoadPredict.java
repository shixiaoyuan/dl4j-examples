package mycode.load;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;


public class LoadPredict {
    private static final int IN_NUM = 4;
    private static final int OUT_NUM = 1;
    private static final int Epochs = 100;

    private static final int lstmLayerSize = 8;

    public static MultiLayerNetwork getNetModel(int nIn, int nOut) {
        //神经网络参数
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(new RmsProp(0.05))
            .seed(12346)
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

    public static void train(MultiLayerNetwork net, LoadIterator iterator){
        //迭代训练次数
        for(int i=1;i<=4;i++) {
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
            //24个时间步长
            for(int j=0;j<24;j++) {
                INDArray output = net.rnnTimeStep(initArray);
                System.out.print(output.getDouble(0)*iterator.getMaxArr()[0]+" ");
            }
            RegressionEvaluation eval = net.evaluateRegression(iterator);
            System.out.println("模型评估："+"\n"+ eval.stats());
            net.rnnClearPreviousState();
        }
        System.out.println("done");

    }

    public static INDArray getInitArray(LoadIterator iter){
        double[] maxNums = iter.getMaxArr();
        INDArray initArray = Nd4j.zeros(1, 4, 1);
        initArray.putScalar(new int[]{0,0,0}, 935.775/maxNums[0]);
        initArray.putScalar(new int[]{0,1,0}, 6/maxNums[1]);
        initArray.putScalar(new int[]{0,2,0}, 4/maxNums[2]);
        initArray.putScalar(new int[]{0,3,0}, 87/maxNums[3]);

        System.out.println("size is "+initArray.size(0)+initArray.size(1)+initArray.size(2));
        return initArray;
    }
    public static void main(String[] args) {
        String inputFile = "/Users/shiyuan/Desktop/essaydata/data.csv";
        int batchSize = 1;
        int exampleLength = 24;
        //初始化深度神经网络
        LoadIterator iterator = new LoadIterator();
        iterator.loadData(inputFile,batchSize,exampleLength);
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        MultiLayerNetwork net = getNetModel(IN_NUM,OUT_NUM);
        int listenerFrequency = 1;
        net.setListeners(new StatsListener(statsStorage, listenerFrequency));
        uiServer.attach(statsStorage);
        train(net, iterator);

    }
}
