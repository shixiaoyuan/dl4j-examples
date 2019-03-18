package mycode.load;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
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
import org.deeplearning4j.util.ModelSerializer;

import java.io.File;
import java.io.IOException;

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
            .l2(0.002)
            .weightInit(WeightInit.XAVIER)
            .list()
            .layer(0, new LSTM.Builder().nIn(nIn).nOut(lstmLayerSize)
                .activation(Activation.TANH).build())
            .layer(1, new LSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
                .activation(Activation.TANH).build())
            .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY)
                .nIn(lstmLayerSize).nOut(nOut).build())
            .backpropType(BackpropType.TruncatedBPTT).tBPTTBackwardLength(12).tBPTTForwardLength(12)
            //.pretrain(false).backprop(true)
            .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));
        return net;
    }

    public static void train(MultiLayerNetwork net, LoadIterator iterator){
        double[] maxNums = iterator.getMaxArr();
        //迭代训练次数
        for(int i=1;i<=Epochs;i++) {
            DataSet dataSet = null;
            while (iterator.hasNext()) {
                dataSet = iterator.next();
                net.fit(dataSet);
            }
            iterator.reset();
            System.out.println();
            System.out.println("=================>完成第"+i+"次完整训练");
            //INDArray initArray = getInitArray(iterator);
            int index =4478;

            //预测连续24个小时负荷值
            System.out.println("真实值：");
            for(int k=0;k<24;++k) {
                INDArray indexArray = Nd4j.zeros(1, 4, 1);
                LoadData loadData = iterator.getdata(index+k);
                indexArray.putScalar(new int[]{0,0,0}, loadData.getLoad()/maxNums[0]);
                indexArray.putScalar(new int[]{0,1,0}, loadData.getTemp()/maxNums[1]);
                indexArray.putScalar(new int[]{0,2,0}, loadData.getDew()/maxNums[2]);
                indexArray.putScalar(new int[]{0,3,0}, loadData.getHum()/maxNums[3]);
                System.out.print(loadData.getLoad()+" ");
            }
            System.out.println();
            System.out.println("预测结果：");
            //24个时间步长
            for(int j=0;j<24;j++) {
                INDArray indexArray = Nd4j.zeros(1, 4, 1);
                LoadData loadData = iterator.getdata(index+j);
                indexArray.putScalar(new int[]{0,0,0}, loadData.getLoad()/maxNums[0]);
                indexArray.putScalar(new int[]{0,1,0}, loadData.getTemp()/maxNums[1]);
                indexArray.putScalar(new int[]{0,2,0}, loadData.getDew()/maxNums[2]);
                indexArray.putScalar(new int[]{0,3,0}, loadData.getHum()/maxNums[3]);
                INDArray output = net.rnnTimeStep(indexArray);
                System.out.print(output.getDouble(0)*maxNums[0]+" ");
            }
            System.out.println();
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
        int batchSize = 3;
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

        //保存模型
        File locationToSave = new File("/Users/shiyuan/Desktop/LSTM.zip");
        boolean saveUpdater = true;
        try {
            ModelSerializer.writeModel(net, locationToSave, saveUpdater);
        } catch (IOException e) {
            e.printStackTrace();
        }
        try {
            MultiLayerNetwork restored = ModelSerializer.restoreMultiLayerNetwork(locationToSave);
            System.out.println("Saved and loaded parameters are equal:      " + net.params().equals(restored.params()));
            System.out.println("Saved and loaded configurations are equal:  " + net.getLayerWiseConfigurations().equals(restored.getLayerWiseConfigurations()));
        } catch (IOException e) {
            e.printStackTrace();
        }



    }
}
