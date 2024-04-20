from .utilis import timeit

from dataclasses import dataclass
from networkx import Graph

import tsplib95

@dataclass
class Instance:
    name: str
    number_of_vertices: int = 0
    number_of_edges: int = 0
    graph: Graph = None

    @timeit
    def get_instance(self):
        instance = tsplib95.load(f"data/{self.name}.tsp")
        instance_graph = instance.get_graph()
        number_of_vertices = instance_graph.number_of_nodes()
        number_of_edges = instance_graph.number_of_edges()
  
        self.number_of_vertices = number_of_vertices
        self.number_of_edges = number_of_edges
        self.graph = instance_graph