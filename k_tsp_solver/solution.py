from k_tsp_solver import Instance
from dataclasses import dataclass
from graphframes import GraphFrame
import math


@dataclass
class Solution():
    instance: Instance
    model: str
    k_factor: float
    path: list
    k_size: int = None
    path_length: int = None

    def __post_init__(self):
        self.k_size = int(math.floor(self.k_factor * self.instance.dimension))

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
        
    # def evaluate(df: pd.DataFrame, population: np.array) -> np.array:
    #     fitness_values = np.zeros(population.shape[0], dtype=int)

    #     for i, solution in enumerate(population):
    #         fitness_values[i] = calculate_path_length(df, solution)

    #     return fitness_values