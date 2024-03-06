from pyspark.sql.functions import date_format, col
from pyspark.sql.functions import window, column, desc, col
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline
staticDataFrame = spark.read.format("csv")
.option("header", "true")
.option("inferSchema", "true")
.load("/data/retail-data/by-day/*.csv")

staticDataFrame.createOrReplaceTempView("retail_data")
staticSchema = staticDataFrame.schema
staticDataFrame
.selectExpr(
"CustomerId",
"(UnitPrice * Quantity) as total_cost",
"InvoiceDate")\.groupBy(
col("CustomerId"), window(col("InvoiceDate"), "1 day"))\
.sum("total_cost")
.show(5)

preppedDataFrame = staticDataFrame
.na.fill(0)
.withColumn("day_of_week", date_format(col("InvoiceDate"), "EEEE"))\
.coalesce(5)

trainDataFrame = preppedDataFrame
.where("InvoiceDate < '2011-07-01'")
testDataFrame = preppedDataFrame
.where("InvoiceDate >= '2011-07-01'")

indexer = StringIndexer()
.setInputCol("day_of_week")
.setOutputCol("day_of_week_index")

encoder = OneHotEncoder()
.setInputCol("day_of_week_index")
.setOutputCol("day_of_week_encoded")

vectorAssembler = VectorAssembler()
.setInputCols(["UnitPrice", "Quantity", "day_of_week_encoded"])\
.setOutputCol("features")

transformationPipeline = Pipeline()
.setStages([indexer, encoder, vectorAssembler])

fittedPipeline = transformationPipeline.fit(trainDataFrame)
transformedTraining = fittedPipeline.transform(trainDataFrame)

kmeans = KMeans()
.setK(20)
.setSeed(1L)
kmModel = kmeans.fit(transformedTraining)
transformedTest = fittedPipeline.transform(testDataFrame)
kmModel.computeCost(transformedTest)
