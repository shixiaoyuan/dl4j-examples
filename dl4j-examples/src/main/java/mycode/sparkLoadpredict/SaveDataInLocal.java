package mycode.sparkLoadpredict;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import mycode.load.LoadIterator;
import java.io.IOException;

import org.apache.hadoop.fs.Path;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.nd4j.linalg.dataset.DataSet;
import java.io.*;


public class SaveDataInLocal {
    public static void main(String[] args) throws IOException {
        LoadIterator iterator = new LoadIterator();
        String inputFile = "/Users/shiyuan/Desktop/essaydata/data.csv";
        int batchSize = 1;
        int exampleLength = 24;
        iterator.loadData(inputFile,batchSize,exampleLength);

        SparkConf sparkConf = new SparkConf();
        sparkConf.setMaster("local[*]");
        sparkConf.setAppName("LSTM Load Predict");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
        Configuration conf = new Configuration();
        conf.addResource(new Path("/Users/shiyuan/hadoop-2.6.5/etc/hadoop/core-site.xml"));
        conf.addResource(new Path("/Users/shiyuan/hadoop-2.6.5/etc/hadoop/hdfs-site.xml"));
        FileSystem fileSystem = FileSystem.get(conf);

        String outputDir = "hdfs://localhost:9000/output/";
        int count = 0;
        while(iterator.hasNext()){
            DataSet ds = iterator.next();
            String filePath = outputDir + "dataset_" + (count++) + ".bin";
            try (OutputStream os = new BufferedOutputStream(fileSystem.create(new Path(filePath)))) {
                ds.save(os);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
