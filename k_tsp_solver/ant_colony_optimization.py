import random
from queue import Queue
from typing import List, Optional
from dataclasses import dataclass, field

import numpy as np

from k_tsp_solver import (
    Instance,
    Solution, 
    Model, 
    ModelName, 
    KFactor, 
    timeit,
    logger
)

random.seed(0)


@dataclass
class AntColonyOptimization(Model):
    name: str = str(ModelName.ANT_COLONY_OPTIMIZATION.value)
    ants: int = 10
    iterations: int = 100
    evaporation_rate: float = 0.1
    alpha: float = 0.7
    beta: float = 0.3
    initial_pheromone: float = 1.0
    best_solution: Optional[Solution] = field(default=None, repr=False)
    best_path_length: int = field(default=None, repr=False)
    is_debugging: bool = field(default=False, repr=False)

    def _update_pheromone(self, pheromone: np.ndarray, solutions: List[Solution]):
        pheromone *= 1 - self.evaporation_rate
        for solution in solutions:
            for i in range(len(solution.path_vertices) - 1):
                current = solution.path_vertices[i]
                next_node = solution.path_vertices[i + 1]
                pheromone[current, next_node] += 1 / solution.path_length
                pheromone[next_node, current] += 1 / solution.path_length
        return pheromone
    
    def _calculate_probability(self, instance: Instance, current: int, next_node: int, pheromone: np.ndarray) -> float:
        pheromone_factor = pheromone[current, next_node] ** self.alpha
        distance_factor = 1 / instance.get_distance(current, next_node) ** self.beta
        return pheromone_factor * distance_factor
    
    def _calculate_probabilities(self, instance: Instance, current: int, visited: np.ndarray, pheromone: np.ndarray) -> np.ndarray:
        unvisited = np.where(visited == False)[0]
        probabilities = np.zeros(instance.number_of_vertices)
        for node in unvisited:
            probabilities[node] = self._calculate_probability(instance, current, node, pheromone)
        return probabilities / np.sum(probabilities)
    
    def _select_next_node(self, instance: Instance, current: int, visited: np.ndarray, pheromone: np.ndarray) -> int:
        probabilities = self._calculate_probabilities(instance, current, visited, pheromone)
        return np.random.choice(instance.number_of_vertices, p=probabilities)
    
    def _generate_solution(
            self, 
            instance: Instance, 
            pheromone: np.ndarray, 
            k_factor: KFactor, 
            has_closed_cycle: bool
        ) -> Solution:
        path = []
        visited = np.zeros(instance.number_of_vertices, dtype=bool)
        first_edge = random.randint(0, instance.number_of_vertices - 1)
        visited[first_edge] = True
        current = first_edge
        path.append(current)

        for _ in range(1, instance.number_of_vertices):
            next_node = self._select_next_node(instance, current, visited, pheromone)
            visited.add(next_node)
            path.append(next_node)
            current = next_node
        
        path.append(0)
        solution = Solution(instance=instance, path_vertices=path, k_factor=k_factor, has_closed_cycle=has_closed_cycle)
        solution.get_path_edges()
        solution.evaluate_edge_path_length()
        path_length = solution.path_length

        if self.best_solution is None or solution.path_length < self.best_path_length:
            self.best_solution = solution
            self.best_path_length = solution.path_length
        return solution
    
    def _generate_solutions(
            self, 
            instance: Instance, 
            pheromone: np.ndarray,  
            k_factor: KFactor, 
            has_closed_cycle: bool
        ) -> List[Solution]:
        solutions = []
        for _ in range(self.ants):
            solution = self._generate_solution(instance, pheromone, k_factor, has_closed_cycle)
            solutions.append(solution)
        return solutions

    # def _debug(self, instance: Instance, pheromone: np.ndarray):
    #     logger.debug(f"pheromone: {pheromone}")
    #     for i in range(self.ants):
    #         solution = self._generate_solution(instance, pheromone)
    #         logger.debug(f"ant {i}: {solution.path} {solution.path_length}")
    #     logger.debug(f"best solution: {self.best_solution.path} {self.best_solution.path_length}")
    #     logger.debug(f"best path length: {self.best_path_length}")
    #     logger.debug("")
    #     return pheromone
    
    @timeit
    def generate_solution(self, instance: Instance, k_factor: KFactor, has_closed_cycle: bool):
        self.best_solution: Solution = None
        self.best_path_length: int = None
        pheromone = np.full((instance.number_of_edges, instance.number_of_edges), self.initial_pheromone)

        for _ in range(self.iterations):
            solutions = self._generate_solutions(instance, pheromone, k_factor, has_closed_cycle)
            self._update_pheromone(pheromone, solutions)
        
        return self.best_solution
