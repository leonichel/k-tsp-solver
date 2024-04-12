from k_tsp_solver import Instance
from dataclasses import dataclass
import math


@dataclass
class Solution():
    instance: Instance
    model: str
    k_factor: float
    path_edges: list = None
    path_vertices: list = None
    path_length: int = None
    k_size: int = None
    

    def __post_init__(self):
        self.k_size = int(math.floor(self.k_factor * self.instance.number_of_vertices))

    # def is_solution_feasible(df: pd.DataFrame, solution: np.array, solution_size: int) -> bool:
    #     if solution.size != solution_size:
    #         return False

    #     edges = df.merge(
    #         pd.DataFrame({'source': solution[:-1], 'target': solution[1:]}),
    #         on=['source', 'target'],
    #         how='inner'
    #     )

    #     trips_number = len(edges)

    #     if trips_number != solution_size - 1:
    #         return False

    #     visited_cities = set(np.concatenate((solution[:-1], solution[1:])))

    #     return len(visited_cities) == solution.size
        
    def evaluate_edge_path_lenght(self):
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