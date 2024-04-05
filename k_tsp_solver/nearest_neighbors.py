from k_tsp_solver import Instance, Solution, Model

from dataclasses import dataclass
from typing import List, Tuple
from pyspark.sql import functions as F

from graphframes import GraphFrame
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import Row


@dataclass
class NearestNeighbors(Model):
    name = "NearestNeighbors"
    parameters = {} 

    def __init__(self):
        self.neighbors_cache = {}

    def get_n_th_shortest_edge(self, instance: Instance, n: int) -> Row:
        shortest_edges = (
            instance.graph.edges
                .filter("src != dst")
                .sort(F.col("weight"))
        )

        # normalize n to avoid “list index out of range” issues
        n = n % shortest_edges.count()

        return shortest_edges.take(n + 1)[n]
    
    def get_neighbors(self, instance: Instance, vertex: int) -> SparkDataFrame:
        if vertex in self.neighbors_cache:
            return self.neighbors_cache[vertex]

        neighbors = (
            instance.graph
            .find("(v1)-[e]->(v2)") 
                .filter(f"""
                    (v1.id = '{vertex}' AND v1.id != v2.id)
                    OR (v2.id = '{vertex}' AND v1.id != v2.id)
                """) 
                .selectExpr(f"CASE WHEN v1.id = '{vertex}' THEN v2.id ELSE v1.id END AS id", "e.weight")
                .distinct()
                .orderBy("e.weight")
        )

        self.neighbors_cache[vertex] = neighbors

        return neighbors
    
    def get_next_vertex(self, neighbors: SparkDataFrame, path: list) -> Row:
        excluded_vertices = ','.join(map(str, path))
        vertex = (
            neighbors
                .filter(f"id NOT IN ({excluded_vertices})")
        ).first()

        return vertex

    
    def generate_solution(self, instance: Instance, k_factor: float, n_solution: int) -> Solution:
        solution = Solution(
            instance=instance,
            model=self.name,
            k_factor=k_factor,
            path=[]
        )

        path = []
        path_length: float = 0.0 

        shortest_edge = self.get_n_th_shortest_edge(instance=instance, n=n_solution)
        first_vertex = shortest_edge.src
        vertex_id = first_vertex
        distance = 0
        
        for _ in range(solution.k_size):
            path.append(vertex_id)
            path_length += float(distance)
            neighbors = self.get_neighbors(instance=instance, vertex=vertex_id)
            vertex = self.get_next_vertex(neighbors=neighbors, path=path)
            vertex_id = vertex.id
            distance = vertex.weight
        
        solution.path = path
        solution.path_length = path_length

        return solution
    
    def generate_multiple_solutions(self, instance: Instance, k_factor: float, n_solutions: int) -> List[Solution]:
        solutions: List[Solution] = []

        for n in range(n_solutions):
            solutions.append(
                self.generate_solution(
                    instance=instance, 
                    k_factor=k_factor, 
                    n_solution=n)
            )

        return solutions