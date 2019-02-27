#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T
import re
from user_agents import parse
# Load PySpark
spark = SparkSession.builder.appName('Analysis').getOrCreate()
sc = pyspark.SparkContext.getOrCreate()
sc.setLogLevel("ERROR")


# Processing & Analytical goals:
# ------------

# In[3]:


# Parsing the raw log file into a RDD
regx = r"^(\S+) (\S+) (\S+) (\S+) (\S+) (\S+) (\S+) (\S+) (\S+) (\S+) (\S+) (\S+) (\S+) \S+ (.*) (\S+) (\S+)$"
rdd = sc.textFile("data/2015_07_22_mktplace_shop_web_log_sample.log.gz").map(lambda x: re.split(regx, x)[1:17])
rdd_ = rdd.map(lambda x: (x[0], x[1], x[2], x[3], float(x[4]), float(x[5]), float(x[6]), int(x[7]), int(x[8]), 
                          int(x[9]), int(x[10]), x[11].strip('"'), x[12].lower(), x[13].strip('"'), x[14], x[15]))


# In[4]:


# Schema for parsed log
Schema = T.StructType([T.StructField("timestamp", T.StringType(), True),
                                    T.StructField("elb", T.StringType(), True),
                                    T.StructField("client_port", T.StringType(), True),
                                    T.StructField("backend_port", T.StringType(), True),
                                    T.StructField("request_processing_time", T.DoubleType(), True),
                                    T.StructField("backend_processing_time", T.DoubleType(), True),
                                    T.StructField("response_processing_time", T.DoubleType(), True),
                                    T.StructField("elb_status_code", T.LongType(), True),
                                    T.StructField("backend_status_code", T.LongType(), True),
                                    T.StructField("received_bytes", T.LongType(), True),
                                    T.StructField("sent_bytes", T.LongType(), True),
                                    T.StructField("request_type", T.StringType(), True), #GET, POST etc.
                                    T.StructField("request", T.StringType(), True),
                                    T.StructField("user_agent", T.StringType(), True),
                                    T.StructField("ssl_cipher", T.StringType(), True),
                                    T.StructField("ssl_protocol", T.StringType(), True)])


# In[5]:


# RDD converted to DataFame
df = spark.createDataFrame(rdd_, schema=Schema).withColumn("client_ip", F.split(F.col("client_port"), ':')[0]).withColumn("unix_timestamp", F.unix_timestamp(F.col("timestamp").substr(0,19).cast('timestamp'))).cache()


# In[6]:


# Printing the summary for the dataframe
df.describe().show(100, False)


# 1) Sessionize the web log by IP. Sessionize = aggregrate all page hits by visitor/IP during a session.
# ---------
# https://en.wikipedia.org/wiki/Session_(web_analytics)

# In[7]:


# UDF to assign session_id to each request.
# IP represents individual User
# Each user session can be of maximum 15 minutes

session_schema = T.ArrayType(T.StructType([T.StructField("client_port", T.StringType(), False),
                                           T.StructField("timestamp", T.StringType(), False),
                                           T.StructField("session_id", T.IntegerType(), False)]))

# UDF to assign session_id to each record
def sessionize(row):
    unix_timestamp = list(list(zip(*row))[0])
    timestamp = list(list(zip(*row))[1])
    client_port = list(list(zip(*row))[2])
    timeframe = 15*60 # Time Frame 900 Seconds
    first_element = unix_timestamp[0]
    current_sess_id = 1
    session_id = [current_sess_id]
    for i in unix_timestamp[1:]:
        if (i - first_element) < timeframe:
            session_id.append(current_sess_id)
        else:
            first_element = i
            current_sess_id += 1
            session_id.append(current_sess_id)
    return zip(client_port, timestamp, session_id)

udf_s = F.udf(lambda x, y, z: sessionize(sorted(zip(z,y,x))), session_schema)


# In[8]:


df_ = df.groupby('client_ip').agg(udf_s(F.collect_list("client_port"), F.collect_list("timestamp"), F.collect_list("unix_timestamp")).alias('session')).select("client_ip", F.explode("session").alias("session")).select("client_ip", F.col("session").client_port.alias("client_port"), F.col("session").timestamp.alias("timestamp"), F.col("session").session_id.alias("session_id"))


# In[9]:


df_.select("client_ip", "client_port", "timestamp", "session_id").show(10, False)


# In[10]:


# Joining the original dataframe with the dataframe with session id, droping the duplicate columns
# Output is a new dataframe = original raw data + session_id
df_1 = df.join(df_, ['client_port', 'timestamp'], 'inner').drop(df_.timestamp).drop(df_.client_ip).drop(df_.client_port).cache()


# 2) Determine the average session time
# ----------

# In[12]:


# Function to extract the session length, number of unique request in a user/IP session
def analyze(df):
    df_ = df.groupby("client_ip", "session_id").agg(F.min("timestamp").alias("from_timestamp"), F.max("timestamp").alias("to_timestamp"), ((F.max("unix_timestamp") - F.min("unix_timestamp") + 1)).alias("session_length"), F.countDistinct("request").alias("unique_url_request"))
    return df_
    


# In[13]:


df_2 = analyze(df_1.select("client_ip", "session_id", "timestamp", "unix_timestamp", "request")).cache()


# In[14]:


df_2.show(10, False)


# In[15]:


print("Number of Distinct Sessions = {}".format(df_2.count()))
print("Average Session Time (Seconds) for all users = {}".format(df_2.agg(F.avg("session_length").alias("avg_session_time")).collect()[0]["avg_session_time"]))


# 3) Determine unique URL visits per session. To clarify, count a hit to a unique URL only once per session.
# ----------------

# In[18]:


# df_2.unique_url_request contains the number of unique URL request made by a client during a session.
print("20 User/IP sessions with most number of unique URL requests")
df_2.select("client_ip", "from_timestamp", "to_timestamp", "session_id", "unique_url_request").sort("unique_url_request", ascending=False).show(20, False)


# 4) Find the most engaged users, ie the IPs with the longest session times.
# ------------
# 

# In[19]:


# df.session_length contains the length of a user session. 
# Aggregate for each user to extract total session time for a user/IP
print("20 Most Engaged Users (IP with largest total session times)")
df_2.groupby("client_ip").agg((F.sum("session_length")/F.lit(60.0)).alias("total_session_length (Minutes)")).sort("total_session_length (Minutes)", ascending=False).show(20, False)


# Additional questions for Machine Learning Engineer (MLE) candidates:
# ---------------

# In[20]:


# Extract the relevant features per Minute 
df_t_1 = df_1.withColumn("hour_minute", df.timestamp.substr(12,5)).withColumn("hour", df.timestamp.substr(12,2).cast('integer')).withColumn("timestamp", F.col("timestamp").substr(0,16).cast('timestamp'))

df_t_2 = df_t_1.groupby("timestamp").agg((F.count("timestamp")/60).alias("label"),
                                        F.sum("received_bytes").alias("received_bytes"), 
                                        F.sum("sent_bytes").alias("sent_bytes"), 
                                        F.countDistinct("request").alias("unique_request_count"), 
                                        F.countDistinct("client_ip").alias("unique_ip_count"), 
                                        F.first("hour_minute").alias("hour_minute"), 
                                        F.first("hour").alias("hour")).sort(["hour_minute"]).cache()


# In[21]:


# Prepare the feature and label, target variable is load per seconds
# Features include the observations from previous minute/timestep
w = Window.partitionBy().orderBy(["hour_minute"])
df_t_3 = df_t_2.select("hour", "hour_minute", F.lag("sent_bytes").over(w).alias("prev_sent_bytes"), 
                       F.lag("received_bytes").over(w).alias("prev_received_bytes"), 
                       F.lag("unique_request_count").over(w).alias("prev_unique_request_count"), 
                       F.lag("unique_ip_count").over(w).alias("prev_unique_ip_count"), 
                       F.lag("label").over(w).alias("prev_load"), "label").where(F.col("prev_sent_bytes").isNotNull()).cache()


# In[22]:


# Splitting the data into train and test set
train, test = df_t_3.randomSplit([0.8, 0.2], seed=2019)


# 1) Predict the expected load (requests/second) in the next minute
# ----------------

# In[25]:


# Using Gradient Boosted Tree for Regression
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=["prev_sent_bytes", "prev_received_bytes", 
                                       "prev_unique_request_count", "prev_unique_ip_count"], outputCol="features")
gbt = GBTRegressor(seed=42, featuresCol="features", labelCol="label")
pipeline = Pipeline(stages=[assembler, gbt])

# Fintuning the tree depth using cross validation
paramGrid = ParamGridBuilder().addGrid(gbt.maxDepth, [4, 5, 6]).build()
crossval = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=RegressionEvaluator(), numFolds=5)

# Run cross-validation, and choose the best set of parameters.
cvModel = crossval.fit(train)

# Make predictions on test documents. cvModel uses the best model found (gbt).
prediction = cvModel.transform(test)


# In[27]:


print("Chosen Parameter value maxDepth for GBT = {}".format(cvModel.bestModel.stages[1]._java_obj.getMaxDepth()))


# In[28]:


# Printing the training and test metric - RMSE, R2
evaluator = RegressionEvaluator()
print("Train")
print("RMSE: %f" % evaluator.evaluate(cvModel.transform(train), {evaluator.metricName: "rmse"}))
print("r2: %f" % evaluator.evaluate(cvModel.transform(train), {evaluator.metricName: "r2"}))
print("")

print("Test")
print("RMSE: %f" % evaluator.evaluate(cvModel.transform(test), {evaluator.metricName: "rmse"}))
print("r2: %f" % evaluator.evaluate(cvModel.transform(test), {evaluator.metricName: "r2"}))


# In[30]:


# Ploting the result on test data
# get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np

plot_data = np.array(sorted(cvModel.transform(test).select(F.col("hour_minute"), F.col("label"), F.col("prediction")).rdd.map(lambda x: [x['hour_minute'], x['label'], x['prediction']]).collect()))

fig = plt.figure(figsize=[10,4])
plt.plot(plot_data[:,0], np.array(plot_data[:,1], 'float'), 'o',
         plot_data[:,0], np.array(plot_data[:,2], 'float'), 'v')
plt.xlabel('Time (HH-MM)')
plt.ylabel('Load (Request/sec)')
plt.xticks(rotation=45)
plt.legend(['Actual', 'Predicted'])
plt.title("Actual vs Predicted load on the test data")
plt.show()


# 2) Predict the session length for a given IP
# ------------

# In[32]:


# UDF to extract the device type based on user_agent string
def get_device_type(user_agent):
    parsed = parse(user_agent)
    return 'bot' if parsed.is_bot else ('mobile' if parsed.is_mobile else 'pc' if parsed.is_pc else 'other')

udf_device_type = F.udf(lambda user_agent: get_device_type(user_agent), T.StringType())


# In[34]:


# Preparing the feature for predicting session length for a given IP based on its historic feature values
df_u_1 = df_t_1.groupby("client_ip", "session_id").agg(((F.max("unix_timestamp") - F.min("unix_timestamp"))/F.lit(60.0)).alias("session_length"),
     F.countDistinct("request").alias("unique_url"), 
     F.sum("sent_bytes").alias("sent_bytes"), 
     F.sum("received_bytes").alias("received_bytes"), 
     udf_device_type(F.first("user_agent")).alias("device_type"),
     F.first("hour").alias("hour")).where(F.col("session_length") > 0)


# In[36]:


# Predict the session length based on the device type, previous session length, hour of the day
w = Window.partitionBy("client_ip").orderBy(["session_id"])
df_u_2 = df_u_1.select("client_ip", "session_id", "session_length", "unique_url", "hour",                    F.lag("sent_bytes").over(w).alias("prev_sent_bytes"),                    F.lag("received_bytes").over(w).alias("prev_received_bytes"),                    F.lag("device_type").over(w).alias("prev_device_type"),                    F.lag("session_length").over(w).alias("prev_session_length"),                    F.lag("unique_url").over(w).alias("prev_unique_url")).where(F.col("prev_device_type").isNotNull()).cache()

train, test = df_u_2.withColumn("label", F.col("session_length")).randomSplit([0.8, 0.2], seed=2019)


# In[ ]:


# Using Gradient Boosted Tree Regressor for predicting the session length
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.feature import OneHotEncoderEstimator

indexer = StringIndexer(inputCol="prev_device_type", outputCol="categoryIndexPrevDeviceType")
encoder = OneHotEncoderEstimator(inputCols=["categoryIndexPrevDeviceType"], outputCols=["categoryDeviceType"])
assembler = VectorAssembler(inputCols=["categoryDeviceType", "prev_session_length", "hour"], outputCol="features")

gbt = GBTRegressor(seed=42, featuresCol="features", labelCol="label")
pipeline = Pipeline(stages=[indexer, encoder, assembler, gbt])

# Finetune the depth of the Gradient Boosted regression tree
paramGrid = ParamGridBuilder().addGrid(gbt.maxDepth, [4, 5, 6]).build()
crossval = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=RegressionEvaluator(), numFolds=5)
# Run cross-validation, and choose the best set of parameters.
cvModel = crossval.fit(train)

# Make predictions on test documents. cvModel uses the best model found (lrModel).
prediction = cvModel.transform(test)


# In[ ]:


print("Parameter value maxDepth for GBTRegressor = {}".format(cvModel.bestModel.stages[3]._java_obj.getMaxDepth()))


# In[ ]:


# Printing the training and test metric - RMSE, R2
evaluator = RegressionEvaluator()
print("Train")
print("RMSE: %f" % evaluator.evaluate(cvModel.transform(train), {evaluator.metricName: "rmse"}))
print("r2: %f" % evaluator.evaluate(cvModel.transform(train), {evaluator.metricName: "r2"}))
print("")
print("Test")
print("RMSE: %f" % evaluator.evaluate(cvModel.transform(test), {evaluator.metricName: "rmse"}))
print("r2: %f" % evaluator.evaluate(cvModel.transform(test), {evaluator.metricName: "r2"}))


# In[ ]:


# Ploting the result on sample test data
# get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np

plot_data = np.array(sorted(cvModel.transform(test).sample(0.01, False).select(F.col("client_ip"), F.col("label"), F.col("prediction")).rdd.map(lambda x: [x['client_ip'], x['label'], x['prediction']]).collect()))

fig = plt.figure(figsize=[10,4])
plt.plot(plot_data[:,0], np.array(plot_data[:,1], 'float'), 'o',
         plot_data[:,0], np.array(plot_data[:,2], 'float'), 'v')
plt.xlabel('client IP')
plt.ylabel('Session Length (Minutes)')
plt.xticks(rotation=45)
plt.legend(['Actual', 'Predicted'])
plt.title("Actual vs Predicted Session Length on the test data")
plt.show()


# 3) Predict the number of unique URL visits by a given IP
# ------------------

# In[ ]:


# Preparing training and test data for predicting number of unique URL visited by a USER/IP during a session
train, test = df_u_2.withColumn("label", F.col("unique_url")).randomSplit([0.8, 0.2], seed=2019)


# In[135]:


# Using a linear regression model to predict the expected number of unique URL visits by a user during a session.
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.feature import OneHotEncoderEstimator, PolynomialExpansion

indexer = StringIndexer(inputCol="prev_device_type", outputCol="categoryIndexPrevDeviceType")
encoder = OneHotEncoderEstimator(inputCols=["categoryIndexPrevDeviceType"], outputCols=["categoryDeviceType"])
assembler = VectorAssembler(inputCols=["prev_session_length", "categoryDeviceType", "prev_unique_url", "hour"], outputCol="vectorized")

polyExpansion = PolynomialExpansion(inputCol="vectorized", outputCol="features")
lr = LinearRegression(featuresCol="features", labelCol="label")
pipeline = Pipeline(stages=[indexer, encoder, assembler, polyExpansion, lr])

# Finetuning the parameters
paramGrid = ParamGridBuilder().addGrid(polyExpansion.degree, [2]).addGrid(lr.regParam, [0.1, 0.01]).addGrid(lr.fitIntercept, [False, True]).build()

crossval = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=RegressionEvaluator(), numFolds=5)
# Run cross-validation, and choose the best set of parameters.
cvModel = crossval.fit(train)

# Make predictions on test documents. cvModel uses the best model found (lrModel).
prediction = cvModel.transform(test)


# In[136]:


print("Best model degree of polynomail = {}".format(cvModel.bestModel.stages[3]._java_obj.getDegree()))
print("Best model regParam value = {}".format(cvModel.bestModel.stages[4]._java_obj.getRegParam()))
print("Best model fitIntercept value = {}".format(cvModel.bestModel.stages[4]._java_obj.getFitIntercept()))


# In[138]:


# Printing the training and test metric - RMSE, R2
evaluator = RegressionEvaluator()
print("Train")
print("RMSE: %f" % evaluator.evaluate(cvModel.transform(train), {evaluator.metricName: "rmse"}))
print("r2: %f" % evaluator.evaluate(cvModel.transform(train), {evaluator.metricName: "r2"}))
print("")
print("Test")
print("RMSE: %f" % evaluator.evaluate(cvModel.transform(test), {evaluator.metricName: "rmse"}))
print("r2: %f" % evaluator.evaluate(cvModel.transform(test), {evaluator.metricName: "r2"}))


# In[139]:


# Ploting the result on sample test data
# get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np

plot_data = np.array(sorted(cvModel.transform(test).sample(0.008, False).select(F.col("client_ip"), F.col("label"), F.col("prediction")).rdd.map(lambda x: [x['client_ip'], x['label'], x['prediction']]).collect()))

fig = plt.figure(figsize=[10,4])
plt.plot(plot_data[:,0], np.array(plot_data[:,1], 'float'), 'o',
         plot_data[:,0], np.array(plot_data[:,2], 'float'), 'v')
plt.xlabel('client IP')
plt.ylabel('Number of Unique URL visited')
plt.xticks(rotation=45)
plt.legend(['Actual', 'Predicted'])
plt.title("Actual vs Predicted unique URL vists on the test data")
plt.show()


# In[ ]:




