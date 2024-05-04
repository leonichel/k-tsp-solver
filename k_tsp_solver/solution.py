import math
from dataclasses import dataclass, field

from k_tsp_solver import (
    Instance, 
    Model, 
    KFactor, 
    dataclass_to_dict
)


@dataclass
class Solution():
    # solution_id: str = field(default_factory=lambda: str(uuid.uuid1()), init=False)
    instance: Instance
    k_factor: KFactor
    model: Model
    path_edges: list = field(default=None, repr=False)
    path_vertices: list = None
    path_length: int = None
    k_size: int = None
    
    def __post_init__(self):
        self.k_size = int(math.floor(self.k_factor.value * self.instance.number_of_vertices))
        
    def evaluate_edge_path_length(self):
        self.path_length = sum(edge[2]["weight"] for edge in self.path_edges)

    def get_path_vertices(self):
        vertices = []

        for edge in self.path_edges:
            source, target, _ = edge

            if source not in vertices:
                vertices.append(source)

            if target not in vertices:
                vertices.append(target)

        self.path_vertices = vertices

    def get_path_edges(self):
        vertices = self.path_vertices
        edges = []

        for i in range(len(vertices) - 1):
            source = vertices[i]
            target = vertices[i + 1]
  
            edge_data = self.instance.graph.get_edge_data(source, target)
            edges.append((source, target, edge_data))

        self.path_edges = edges

    def get_solution_as_dict(self) -> dict:   
        solution = dataclass_to_dict(self)
        solution["instance"] = dataclass_to_dict(solution["instance"])
        solution["model"] = str(dataclass_to_dict(solution["model"]))

        return solution