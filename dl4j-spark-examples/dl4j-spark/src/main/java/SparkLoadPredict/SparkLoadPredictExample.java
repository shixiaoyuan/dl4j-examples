package SparkLoadPredict;

import com.beust.jcommander.Parameter;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import scala.Tuple2;


import java.util.*;

public class SparkLoadPredictExample {
    private static LoadIterator iterator = new LoadIterator();
    private static LoadPredict loadPredict = new LoadPredict();
    private static final Logger log = LoggerFactory.getLogger(SparkLoadPredictExample.class);
    @Parameter(names = "-useSparkLocal", description = "Use spark local (helper for testing/running without spark submit)", arity = 1)
    private static boolean useSparkLocal = true;

    @Parameter(names = "-batchSizePerWorker", description = "Number of examples to fit each worker with")
    private int batchSizePerWorker = 8;   //How many examples should be used per worker (executor) when fitting?

    @Parameter(names = "-numEpochs", description = "Number of epochs for training")
    public static void main(String[] args) throws Exception {
        int IN_NUM = 4;
        int OUT_NUM = 1;
        int lstmLayerSize = 8;

        //Set up network configuration:
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(new RmsProp(0.05))
            .seed(12346)
            .l2(0.001)
            .weightInit(WeightInit.XAVIER)
            .list()
            .layer(0, new LSTM.Builder().nIn(IN_NUM).nOut(lstmLayerSize)
                .activation(Activation.TANH).build())
            .layer(1, new LSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
                .activation(Activation.TANH).build())
            .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY)
                .nIn(lstmLayerSize).nOut(OUT_NUM).build())
            .pretrain(false).backprop(true)
            .build();



        int averagingFrequency = 3;

        //Set up Spark configuration and context
        SparkConf sparkConf = new SparkConf();
        if (useSparkLocal) {
            sparkConf.setMaster("local[*]");
        }
        sparkConf.setAppName("LSTM Character Example");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);


        //Create the TrainingMaster instance
        int examplesPerDataSetObject = 1;
        ParameterAveragingTrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(examplesPerDataSetObject)
            .workerPrefetchNumBatches(2)    //Asynchronously prefetch up to 2 batches
            .averagingFrequency(averagingFrequency)
            .batchSizePerWorker(1)
            .build();

        //Create the SparkDl4jMultiLayer instance and fit the network using the training data
        SparkDl4jMultiLayer sparkNetwork = new SparkDl4jMultiLayer(sc, conf, tm);
        sparkNetwork.setListeners(new ScoreIterationListener(1));

        String inputFile = "/Users/shiyuan/Desktop/essaydata/data.csv";
        int batchSize = 1;
        int exampleLength = 24;
        iterator.loadData(inputFile,batchSize,exampleLength);


        List<DataSet> dataSetList = new ArrayList<>();
        JavaRDD<DataSet> dataSet = null;
        while (iterator.hasNext()) {
            dataSetList.add(iterator.next());
        }
        dataSet = sc.parallelize(dataSetList);

        //Do training
        for(int i=0;i<2;++i) {
            sparkNetwork.fit(dataSet);
        }

        //Delete the temp training files, now that we are done with them
        tm.deleteTempFiles(sc);
        System.out.println("predict successfully!");

//        INDArray indArray = loadPredict.getInitArray(iterator);
//        List<Tuple2<Integer,INDArray>> lt = new ArrayList<>();
//        Tuple2<Integer,INDArray> tuple2 = new Tuple2<>(1,indArray);
//        lt.add(tuple2);
//        JavaPairRDD<Integer,INDArray> data = sc.parallelizePairs(lt);

//        JavaRDD<INDArray> data = sc.parallelize(Arrays.asList(indArray));
//        JavaPairRDD<Integer,INDArray> line = data.mapToPair(x -> {return new Tuple2(1,x);});

//        JavaPairRDD<Integer, INDArray> line = data.mapToPair(new PairFunction<INDArray, Integer, INDArray>() {
//            @Override
//            public Tuple2<Integer, INDArray> call(DataSet dataSet) throws Exception {
//                return new Tuple2<>(1,indArray);
//            }
//        });**/
        INDArray indArray = loadPredict.getInitArray(iterator);
        SparkSession spark= SparkSession.builder().appName("").getOrCreate();
        JavaSparkContext jsc = new JavaSparkContext(spark.sparkContext());
        JavaRDD<INDArray> data = jsc.parallelize(Arrays.asList(indArray));
        JavaPairRDD<Integer,INDArray> line = data.mapToPair(x -> {return new Tuple2(1,x);});
        System.out.println(sparkNetwork.feedForwardWithKey(line,1));



    }



    /**public static JavaRDD<DataSet> getTrainingData(JavaSparkContext sc) throws IOException {
     ArrayList<Integer> list = new ArrayList<>();
     for(int i=0;i<iterator.getSize();++i) {
     list.add(1);
     }
     JavaRDD<Integer> rawStrings = sc.parallelize(list);
     return rawStrings.map(new IntegerToDataSetFn(iterator));
     }

     private static class IntegerToDataSetFn implements Function<Integer, DataSet> {
     private final LoadIterator iterator;

     private IntegerToDataSetFn(LoadIterator iterator) {
     this.iterator = iterator;
     }

     @Override
     public DataSet call(Integer s) throws Exception {
     return iterator.next();
     }
     }**/


}



