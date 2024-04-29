import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class WineQualityTraining {
    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("WineQualityTraining");
        try (JavaSparkContext jsc = new JavaSparkContext(conf)) {
            SparkSession spark = SparkSession.builder().appName("WineQualityTraining").getOrCreate();

            Dataset<Row> trainingData = spark.read().format("csv")
                    .option("header", "true")
                    .option("inferSchema", "true")
                    .load("s3a://your-s3-bucket/TrainingDataset.csv");

            String[] featureColumns = {"fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
                    "pH", "sulphates", "alcohol"};
            VectorAssembler assembler = new VectorAssembler()
                    .setInputCols(featureColumns)
                    .setOutputCol("features");

            Dataset<Row>[] splits = trainingData.randomSplit(new double[]{0.8, 0.2}, 42);
            Dataset<Row> trainData = splits[0];
            Dataset<Row> validationData = splits[1];

            LogisticRegression lr = new LogisticRegression()
                    .setMaxIter(10)
                    .setRegParam(0.3)
                    .setElasticNetParam(0.8)
                    .setFamily("multinomial")
                    .setLabelCol("quality");

            Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{assembler, lr});
            PipelineModel model = pipeline.fit(trainData);

            Dataset<Row> predictions = model.transform(validationData);
            MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                    .setLabelCol("quality")
                    .setPredictionCol("prediction")
                    .setMetricName("f1");
            double f1Score = evaluator.evaluate(predictions);
            System.out.println("F1 score on validation data: " + f1Score);

            model.write().overwrite().save("s3a://your-s3-bucket/wineQualityModel");

            spark.stop();
        }
    }
}
