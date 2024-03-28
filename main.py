# %%

from k_tsp_solver import Instance, GeneticAlgorithm
from pyspark.sql import SparkSession

from pyspark.sql import functions as F

# %%

spark = (
    SparkSession
    .builder
    .appName('GraphFrames_Test')
    .config("spark.jars.packages", "graphframes:graphframes:0.8.3-spark3.5-s_2.12")
    .config("spark.jars.repositories", "https://repos.spark-packages.org")
    .getOrCreate()
)

# %%

instance = Instance.get_instance(
    spark, 
    "burma14.tsp"
)

# %%

instance.graph.vertices

# %%
