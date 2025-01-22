from typing import List
from functools import lru_cache
from dataclasses import dataclass

from k_tsp_solver import (
    Instance, 
    Solution, 
    Model, 
    ModelName, 
    KFactor, 
    timeit
)


@dataclass(
    unsafe_hash=True
)
class NearestNeighbors(Model):
    name: str = str(ModelName.NEAREST_NEIGHBORS.value)

    @lru_cache
    def sort_edges_by_weight(self, instance: Instance) -> list:        
        edges = [(u, v, d) for u, v, d in instance.graph.edges(data=True) if u != v]
        sorted_edges = sorted(edges, key=lambda x: x[2]['weight'])

        return sorted_edges

    def get_n_th_shortest_edge(self, sorted_edges: list, n: int) -> tuple:
        n = n % len(sorted_edges)   # normalize n to avoid “list index out of range” issues

        return sorted_edges[n]
    
    @lru_cache
    def get_neighbors(self, instance: Instance, vertex: int) -> list:
        neighbors = [
            (vertex, neighbor, edge_data)
            for neighbor, edge_data in instance.graph.adj[vertex].items()
            if neighbor != vertex
        ]
    
        return neighbors
    
    def filter_unvisited_neighbors(self, neighbors: list, visited: set) -> list:
        return [neighbor for neighbor in neighbors if neighbor[1] not in visited]
    
    def get_next_vertex(self, neighbors: list) -> list:
        return min(neighbors, key=lambda edge: edge[2]['weight'])
    
    def get_edge_to_close_cycle(self, instance: Instance, path_edges: list) -> tuple:
        first_vertex = path_edges[0][0]
        last_vertex = path_edges[-1][0]
        return last_vertex, first_vertex, instance.graph[last_vertex][first_vertex]
    
    def append_edge_to_path(self, path_edges: list, visited: list, edge: tuple) -> None:
        path_edges.append(edge)
        visited.add(edge[0])
        visited.add(edge[1])
    
    def generate_solution(
        self, 
        instance: Instance, 
        k_factor: KFactor, 
        has_closed_cycle: bool,
        n_solution: int = 0
    ) -> Solution:
        path_edges = []
        visited = set()
        last_vertex: int = 0

        solution = Solution(
            instance=instance,
            model=self,
            k_factor=k_factor,
            has_closed_cycle=has_closed_cycle,
            path_edges=path_edges
        )

        sorted_edges = self.sort_edges_by_weight(instance=instance)
        shortest_edge = self.get_n_th_shortest_edge(sorted_edges=sorted_edges, n=n_solution)
        self.append_edge_to_path(path_edges=path_edges, visited=visited, edge=shortest_edge)
        last_vertex = path_edges[-1][1]
        vertices_to_append = solution.k_size - 3 if has_closed_cycle else solution.k_size - 2

        for _ in range(vertices_to_append):
            all_neighbor_edges = self.get_neighbors(instance=instance, vertex=last_vertex)
            unvisited_neighbor_edges = self.filter_unvisited_neighbors(neighbors=all_neighbor_edges, visited=visited)
            shortest_edge = self.get_next_vertex(neighbors=unvisited_neighbor_edges)
            self.append_edge_to_path(path_edges=path_edges, visited=visited, edge=shortest_edge)
            last_vertex = path_edges[-1][1]
        
        if has_closed_cycle:
            edge_to_close_cycle = self.get_edge_to_close_cycle(instance=instance, path_edges=path_edges)
            self.append_edge_to_path(path_edges=path_edges, visited=visited, edge=edge_to_close_cycle)

        solution.evaluate_edge_path_length()
        solution.get_path_vertices()

        return solution
    
    @timeit
    def generate_multiple_solutions(
        self, 
        instance: Instance, 
        k_factor: KFactor, 
        has_closed_cycle: bool,
        n_solutions: int
    ) -> List[Solution]:
        solutions: List[Solution] = []

        for n in range(n_solutions):
            solution = self.generate_solution(
                instance=instance, 
                k_factor=k_factor, 
                has_closed_cycle=has_closed_cycle,
                n_solution=n
            )
            solutions.append(solution)

        return solutions