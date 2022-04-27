# from IPython.display import display
# import pandas as pd
from pyspark.mllib.linalg import Vectors
from pyspark.ml.feature import VectorAssembler, IndexToString, StringIndexer, VectorIndexer, MinMaxScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.sql.types import DoubleType
from pyspark.sql.session import SparkSession
# from pyspark.sql.functions import *
from pyspark.context import SparkContext

# sc = SparkContext('local')
spark = SparkSession.builder.appName('Training').getOrCreate()

# TrainingDataset = pd.read_csv("TrainingDataset.csv", sep=';')
# result = TrainingDataset.head(10)
# print("First 10 rows of the DataFrame:")
# print(result)

TrainingDF = spark.read.csv('s3a://parth643assignment2/TrainingDataset.csv',header='true', inferSchema='true', sep=';')
ValidationDF = spark.read.csv('s3a://parth643assignment2/ValidationDataset.csv',header='true', inferSchema='true', sep=';')
trainRFDF = TrainingDF
validRFDF = ValidationDF

# select the columns to be used as the features (all except `quality`)
featureColumnsT = [c for c in TrainingDF.columns if c != 'quality']
featureColumnsV = [c for c in ValidationDF.columns if c != 'quality']


# create and configure the assemblers
assemblerT = VectorAssembler(inputCols=featureColumnsT, outputCol="features")
assemblerV = VectorAssembler(inputCols=featureColumnsV, outputCol="features")

# transform the original data
trainDataDF = assemblerT.transform(TrainingDF)
validDataDF = assemblerV.transform(ValidationDF)

# display(trainDataDF.limit(3))
# display(validDataDF.limit(3))

# Linear Regression
print('------- Linear Regression -------')

# Linear Regression
# fit a `LinearRegression` model using features in colum `features` and label in column `quality`
lr = LinearRegression(maxIter=30, regParam=0.3, elasticNetParam=0.3, featuresCol="features", labelCol="quality")

lrModel = lr.fit(trainDataDF)

lrPrediction = lrModel.transform(validDataDF)
lrPrediction.select('prediction','quality').show(5)

lrModel.write().overwrite().save("s3a://parth643assignment2/linear.model")

lrEvaluator = RegressionEvaluator(labelCol = 'quality',
                                   predictionCol='prediction',
                                   metricName="rmse")
rmse = lrEvaluator.evaluate(lrPrediction)


lrEvaluator2= RegressionEvaluator(labelCol = 'quality',
                                   predictionCol='prediction',
                                   metricName="r2")
r2 = lrEvaluator2.evaluate(lrPrediction)

print("RMS=%g" % rmse)
print("R squared = ", r2)

print('=======================================================')
print('#######################################################')
print('=======================================================')

print('------- Random Forest -------')

totalColumns = trainRFDF.columns

from pyspark.sql.functions import udf
stages = []
unlist = udf(lambda x: round(float(list(x)[0]),3), DoubleType())

for columnName in totalColumns[:-1]:
    stages = []
    vectorAssembler = VectorAssembler(inputCols=[columnName],outputCol=columnName+'_vectorized')
    stages.append(vectorAssembler)
    stages.append(MinMaxScaler(inputCol=columnName+'_vectorized', outputCol=columnName+'_scaled'))
    pipeline = Pipeline(stages=stages)
    trainRFDF = pipeline.fit(trainRFDF).transform(trainRFDF).withColumn(
        columnName+"_scaled", unlist(columnName+"_scaled")).drop(columnName+"_vectorized").drop(columnName)

# trainRFDF.show(5)

for columnName in totalColumns[:-1]:
    stages = []
    vectorAssembler = VectorAssembler(inputCols=[columnName],outputCol=columnName+'_vectorized')
    stages.append(vectorAssembler)
    stages.append(MinMaxScaler(inputCol=columnName+'_vectorized', outputCol=columnName+'_scaled'))
    pipeline = Pipeline(stages=stages)
    validRFDF = pipeline.fit(validRFDF).transform(validRFDF).withColumn(
        columnName+"_scaled", unlist(columnName+"_scaled")).drop(columnName+"_vectorized").drop(columnName)

# validRFDF.show(5)

vectorAssembler = VectorAssembler(
    inputCols=[columnName+"_scaled" for columnName in totalColumns[:-1]],
    outputCol='features')

labelIndexer = StringIndexer(inputCol=totalColumns[-1], outputCol="indexedLabel").fit(trainRFDF)

rf = RandomForestClassifier(labelCol='indexedLabel', featuresCol="features", numTrees=100)

labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)

pipeline = Pipeline(stages=[labelIndexer, vectorAssembler, rf, labelConverter])

model = pipeline.fit(trainRFDF)
model.write().overwrite().save("s3a://parth643assignment2/rfc.model")

predictions = model.transform(validRFDF)
predictions.select("predictedLabel", totalColumns[-1], "features").show(5)

evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
rfModel = model.stages[2]

print("Error:" )
print((1.0 - accuracy))
print("Model Summary:") 
print(rfModel) 
print("Accuracy:") 
print(accuracy)

print('******* END OF SCRIPT *********')