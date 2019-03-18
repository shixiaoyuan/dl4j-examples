package mycode.load;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;

public class LoadIterator  implements DataSetIterator {


    private static final int VECTOR_SIZE = 4;
    //每批次的训练数据组数
    private int batchNum;

    //每组训练数据长度
    private int exampleLength;

    //数据集
    private List<LoadData> dataList;

    //存放剩余数据组的index信息
    private List<Integer> dataRecord;

    private double[] maxNum;

    //构造方法
    public LoadIterator(){
        dataRecord = new ArrayList<>();
    }

    public int getSize() {
        return dataList.size();
    }

    //加载数据并初始化
    public boolean loadData(String fileName, int batchNum, int exampleLength){
        this.batchNum = batchNum;
        this.exampleLength = exampleLength;
        maxNum = new double[4];
        try {
            readDataFromFile(fileName);
        }catch (Exception e){
            e.printStackTrace();
            return false;
        }
        //重置训练批次列表
        resetDataRecord();
        return true;
    }

    /**
     * 重置训练批次列表
     * */
    private void resetDataRecord(){
        dataRecord.clear();
        int total = dataList.size()/exampleLength+1;
        for( int i=0; i<total; i++ ){
            dataRecord.add(i * exampleLength);
        }
    }
    /**
     * 从文件中读取负载数据
     * */
    public List<LoadData> readDataFromFile(String fileName) throws IOException{
        for(int i=0;i<maxNum.length;i++){
            maxNum[i] = 0;
        }

        dataList = new ArrayList<>();
        FileInputStream fis = new FileInputStream(fileName);
        BufferedReader in = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
        String line = in.readLine();

        System.out.println("读取数据..");
        while(line!=null){
            String[] strArr = line.split(",");
            if(strArr.length>=5) {
                //每小时的数据是一个LoadData
                LoadData data  = new LoadData();
                //一个nums数组存放的是一小时的数据
                double[] nums = new double[4];
                for(int j=0;j<4;j++){
                    //读取1-4列数据
                    nums[j] = Double.valueOf(strArr[j+1]);
                    if( nums[j]>maxNum[j] ){
                        maxNum[j] = nums[j];
                    }
                }
                //构造data对象
                data.setLoad(Double.valueOf(nums[0]));
                data.setTemp(Double.valueOf(nums[1]));
                data.setDew(Double.valueOf(nums[2]));
                data.setHum(Double.valueOf(nums[3]));
                dataList.add(data);

            }
            line = in.readLine();
        }
        in.close();
        fis.close();
        return dataList;
    }

    public double[] getMaxArr(){
        return this.maxNum;
    }

    public void reset(){
        resetDataRecord();
    }

    public boolean hasNext(){
        return dataRecord.size() > 0;
    }

    //batchNum恒为1
    public DataSet next(){
        return next(batchNum);
    }

    /**
     * 获得接下来一次的训练数据集
     * */
    public DataSet next(int num){
        if( dataRecord.size() <= 0 ) {
            throw new NoSuchElementException();
        }
        int actualBatchSize = Math.min(num, dataRecord.size());
        int actualLength = Math.min(exampleLength,dataList.size()-dataRecord.get(0));
        //{实际批数，特征向量的维数，实际长度}
        INDArray input = Nd4j.create(new int[]{actualBatchSize,VECTOR_SIZE,actualLength}, 'f');
        INDArray label = Nd4j.create(new int[]{actualBatchSize,1,actualLength}, 'f');
        LoadData nextData = null,curData = null;

        //获取每批次的训练数据和标签数据,一次循环一个完整时间步长
        for(int i=0;i<actualBatchSize;i++){
            int index = dataRecord.remove(0);
            int endIndex = Math.min(index+exampleLength,dataList.size()-1);
            curData = dataList.get(index);
            for(int j=index;j<endIndex;j++){
                //获取数据信息
                nextData = dataList.get(j+1);
                //构造训练向量
                int c = j-index;
                input.putScalar(new int[]{i, 0, c}, curData.getLoad()/maxNum[0]);
                input.putScalar(new int[]{i, 1, c}, curData.getTemp()/maxNum[1]);
                input.putScalar(new int[]{i, 2, c}, curData.getDew()/maxNum[2]);
                input.putScalar(new int[]{i, 3, c}, curData.getHum()/maxNum[3]);

                //构造label向量
                label.putScalar(new int[]{i, 0, c}, nextData.getLoad()/maxNum[0]);
                curData = nextData;
            }
            if(dataRecord.size()<=0)
                break;
        }

        return new DataSet(input, label);
    }

    public int batch() {
        return batchNum;
    }

    public int cursor() {
        return totalExamples() - dataRecord.size();
    }

    public int numExamples() {
        return totalExamples();
    }

    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public int totalExamples() {
        return (dataList.size()) / exampleLength;
    }

    public int inputColumns() {
        return dataList.size();
    }

    public int totalOutcomes() {
        return 1;
    }

    @Override
    public List<String> getLabels() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException();
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        throw new UnsupportedOperationException("Not implemented");
    }

    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    public LoadData getdata(int index) {
        return dataList.get(index);
    }
}
