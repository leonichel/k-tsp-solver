# from k_tsp_solver import Instance, Solution, Model

# from dataclasses import dataclass
# from typing import List

# from pyspark.sql import functions as F
# from pyspark.sql.types import StructType, StructField, IntegerType

# from pyspark.sql import DataFrame as SparkDataFrame
# from pyspark.sql import SparkSession
# from pyspark.sql import Row


# @dataclass
# class SparkNearestNeighbors(Model):
#     name = "NearestNeighbors"
#     parameters = {} 

#     def __init__(self):
#         self.neighbors_cache = {}

#     def sort_edges_by_distance(self, instance: Instance) -> SparkDataFrame:
#         return (
#             instance.graph
#                 .filter("src != dst")
#                 .sort(F.col("weight"))
#         ).cache()

#     def get_n_th_shortest_edge(self, instance: Instance, shortest_edges: SparkDataFrame, n: int) -> SparkDataFrame:
#         n = n % instance.number_of_edges    # normalize n to avoid “list index out of range” issues

#         return shortest_edges.offset(n).limit(1)
    
#     def get_neighbors(self, instance: Instance, vertex: SparkDataFrame) -> SparkDataFrame:
#         return (
#             instance.graph.alias("graph")
#                 .join(
#                     other=F.broadcast(vertex).alias("vertex"),
#                     on=(F.col("graph.src") == F.col("vertex.dst")),
#                     how="inner"
#                 )
#                 .filter(F.col("graph.src") != F.col("graph.dst"))
#                 .select("graph.src", "graph.dst", "graph.weight")
#         )
    
#     def get_next_vertex(self, neighbors: SparkDataFrame, path: SparkDataFrame) -> SparkDataFrame:
#         return (
#             neighbors.alias("neighbors")
#                 .join(
#                     other=F.broadcast(path).alias("path"),
#                     on=(
#                         (F.col("neighbors.dst") == F.col("path.src"))
#                     ),
#                     how="left_anti"
#                 )
#                 .select("neighbors.src", "neighbors.dst", "neighbors.weight")
#                 .orderBy("neighbors.weight")
#                 .limit(1)
#         )
    
#     def create_empty_path(self, spark: SparkSession):
#         schema = StructType([
#             StructField("id",  IntegerType(), False),
#             StructField("src", IntegerType(), False),
#             StructField("dst", IntegerType(), False),
#             StructField("weight", IntegerType(), False)
#         ])

#         return spark.createDataFrame([], schema)
    
#     def append_edge_to_path(self, path: SparkDataFrame, edge: Row, id: int) -> SparkDataFrame:
#         path = path.union(
#             edge
#                 .withColumn("id", F.lit(id))
#                 .select(["id", "src", "dst", "weight"])
#         )

    
#     def generate_solution(self, spark: SparkSession, instance: Instance, k_factor: float, n_solution: int) -> Solution:
#         solution = Solution(
#             instance=instance,
#             model=self.name,
#             k_factor=k_factor,
#             path=[]
#         )

#         shortest_edges = self.sort_edges_by_distance(instance=instance)

#         shortest_edge = self.get_n_th_shortest_edge(
#             instance=instance, 
#             shortest_edges=shortest_edges,
#             n=n_solution
#         )

#         path = self.create_empty_path(spark=spark)

#         for i in range(solution.k_size):
#             path = self.append_edge_to_path(path=path, edge=shortest_edge, id=i)
#             path_last_vertex = path.select("dst").orderBy(F.desc("id")).limit(1)
#             neighbors = self.get_neighbors(instance=instance, vertex=path_last_vertex)
#             shortest_edge = self.get_next_vertex(neighbors=neighbors, path=path)
    
#     def generate_multiple_solutions(self, instance: Instance, k_factor: float, n_solutions: int) -> List[Solution]:
#         solutions: List[Solution] = []

#         for n in range(n_solutions):
#             solutions.append(
#                 self.generate_solution(
#                     instance=instance, 
#                     k_factor=k_factor, 
#                     n_solution=n)
#             )

#         return solutions