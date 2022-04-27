import pyspark
import argparse
from pyspark.ml.regression import LinearRegressionModel, LinearRegression
from pyspark.sql import SparkSession
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml import PipelineModel, Pipeline
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler                    
import pyspark.sql.functions as func
from pyspark.mllib.evaluation import MulticlassMetrics


spark = SparkSession.builder.appName('Prediction').getOrCreate()

def preprocess(df, classifier):
    totalColumns = df.columns
    df = df.select(*(col(c).cast("double").alias(c) for c in df.columns))
    stages = []
    unlist = udf(lambda x: round(float(list(x)[0]),3), DoubleType())
    for columnName in totalColumns[:-1]:
        stages = []
        vectorAssembler = VectorAssembler(inputCols=[columnName],outputCol=columnName+'_vectorized')
        stages.append(vectorAssembler)
        stages.append(MinMaxScaler(inputCol=columnName+'_vectorized', outputCol=columnName+'_scaled'))
        pipeline = Pipeline(stages=stages)
        df = pipeline.fit(df).transform(df).withColumn(
            columnName+"_scaled", unlist(columnName+"_scaled")).drop(
            columnName+"_vectorized").drop(columnName)
    # if you want to run linear, don't comment next 2 lines
    if classifier == 'linear':
        vectorAssembler = VectorAssembler(
            inputCols=[columnName+'_scaled' for columnName in totalColumns[:-1]],
            outputCol='features')
        df = vectorAssembler.transform(df)
    return df, totalColumns

def run(testCsv, classifier):
    print(testCsv)
    print(classifier)
    df = spark.read.format("com.databricks.spark.csv").csv(testCsv, header=True, sep=";")
    df, totalColumns = preprocess(df, classifier)
    if classifier == 'linear':
        model = LinearRegressionModel.load("s3a://parth643assignment2/linear.model")
    else:
        model = PipelineModel.load("s3a://parth643assignment2/rfc.model")
    df = model.transform(df)
    if classifier =='linear':
        df = df.withColumn("predictionWithRound", func.round(df["prediction"], 0)).drop('prediction')
        df = df.select("predictionWithRound", totalColumns[-1])
    return df, totalColumns

def f1Result(df, totalColumns, classifier):
    label_column = totalColumns[-1]
    if classifier == 'linear':
        predictionAndLabels = df.select(['predictionWithRound', totalColumns[-1]])
    else:
        predictionAndLabels = df.select(['predictedLabel', totalColumns[-1]])
    labels = df.select([label_column]).distinct()
    header = labels.rdd.first()
    labels = labels.rdd.filter(lambda line: line !=header)
    print("QWERTY+++++++++++++++++++++++++")
    print(predictionAndLabels)
    print("+++++++++++++++++++++++++")
    header = predictionAndLabels.rdd.first()
    tempPredictionAndLabels = predictionAndLabels.rdd.filter(lambda line: line != header)
    tempPredictionAndLabel = tempPredictionAndLabels.map(lambda lp: (float(lp[0]), float(lp[1])))
    metrics = MulticlassMetrics(tempPredictionAndLabel)
    # Overall statistics
    precision = metrics.precision()
    recall = metrics.recall()
    f1Score = metrics.fMeasure()
    print("Results:")
    print("Precision:")
    print(precision)
    print("Recall:")
    print(recall)
    print("F1 Score:" )
    print(f1Score)

import argparse
parser = argparse.ArgumentParser(description='Prediction')
parser.add_argument('--testCsv', required=True, help='Provide S3 file path')
parser.add_argument('--classifier', required=True, help='Provide a classifier linear or rfc')
args = parser.parse_args()
print("args")
print(args)
df, totalColumns = run(args.testCsv, args.classifier)
f1Result(df, totalColumns, args.classifier)
