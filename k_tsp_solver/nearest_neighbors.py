from k_tsp_solver import Instance, Solution, Model

from dataclasses import dataclass
from pyspark.sql import functions as F

from graphframes import GraphFrame
from pyspark.sql import Row


@dataclass
class NearestNeighbors(Model):
    name = "NearestNeighbors"
    parameters = {} 

    def get_n_th_shortest_edge(instance: Instance, n: int) -> Row:
        graph = instance.graph
        n_th_edge = (
            graph.edges
                .filter("src != dst")
                .sort(F.col("weight"))
        ).collect()[n]

        return n_th_edge
    
    def get_neighbors(instance: Instance, vertice: int) -> GraphFrame:
        graph = instance.graph
        neighbors = (
            graph.find("(v1)-[]->(v2)") 
                .filter(f"v1.id = '{vertice}' OR v2.id = '{vertice}'") 
                .selectExpr("CASE WHEN v1.id = '{0}' THEN v2.id ELSE v1.id END AS connected_vertex".format(vertice))
                .distinct()
        )

        return neighbors

    # def get_nearest_city(neighbors: np.array) -> int:
    #     if neighbors.size == 0:
    #         return None

    #     nearest_index = neighbors['weight'].idxmin()
    #     nearest_city = neighbors.loc[nearest_index, 'target']

    #     return nearest_city