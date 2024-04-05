# %%

from k_tsp_solver import Instance, GeneticAlgorithm, NearestNeighbors, Solution
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

instance = Instance(
    "dsj1000.tsp"
)

instance.get_instance(spark=spark)

# %%
instance.name, instance.dimension

# %%

nearest_neighbors = NearestNeighbors()

# %%

nearest_neighbors.generate_solution(instance=instance, k_factor=1/3, n_solution=0)

# %%

nearest_neighbors.generate_multiple_solutions(instance=instance, k_factor=1/3, n_solutions=100)
# %%
