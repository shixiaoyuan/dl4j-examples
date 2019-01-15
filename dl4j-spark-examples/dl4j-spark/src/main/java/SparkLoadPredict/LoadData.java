package SparkLoadPredict;

import lombok.Data;

@Data
public class LoadData {
    private double load;
    private double temp;
    private double dew;
    private double hum;
}
