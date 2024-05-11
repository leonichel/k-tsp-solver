from dataclasses import dataclass, field

import tsplib95
from networkx import Graph

from k_tsp_solver import timeit

@dataclass(
    unsafe_hash=True
)
class Instance:
    name: str
    number_of_vertices: int = 0
    number_of_edges: int = 0
    symmetrical_type: str = None
    graph: Graph = field(default=None, repr=False)

    @timeit
    def get_instance(self):
        instance = tsplib95.load(f"data/{self.name}")
        instance_graph = instance.get_graph()
        number_of_vertices = instance_graph.number_of_nodes()
        number_of_edges = instance_graph.number_of_edges()
  
        self.number_of_vertices = number_of_vertices
        self.number_of_edges = number_of_edges
        self.symmetrical_type = instance.type
        self.graph = instance_graph