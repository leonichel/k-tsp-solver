from dataclasses import dataclass
import tsplib95
import networkx as nx
from graphframes import GraphFrame
from pyspark.sql import SparkSession


@dataclass
class Instance:
    name: str
    dimension: int
    graph: GraphFrame

    @classmethod
    def get_instance(cls, spark: SparkSession, instance_name: str) -> 'Instance':
        instance = tsplib95.load(f"data/{instance_name}")
        instance_name = instance.name
        instance_dimension = instance.dimension

        graph_data = nx.to_dict_of_dicts(instance.get_graph())

        vertices_list = []
        for key, _ in graph_data.items():
            vertices_list.append((key,))
        vertices_df = spark.createDataFrame(vertices_list, ["id"])

        edges_list = []
        for source_id, edges_data in graph_data.items():
            for target_id, edge_info in edges_data.items():
                edges_list.append((source_id, target_id, edge_info['weight']))
        edges_df = spark.createDataFrame(edges_list, ["src", "dst", "weight"])

        instance_graph = GraphFrame(vertices_df, edges_df)

        return cls(
            name=instance_name, 
            dimension=instance_dimension, 
            graph=instance_graph
        )