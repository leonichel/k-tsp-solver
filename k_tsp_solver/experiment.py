from k_tsp_solver import Instance, NearestNeighbors, GeneticAlgorithm, Solution, ModelName, KFactor

from itertools import count
from dataclasses import dataclass, field
from typing import List


@dataclass
class Experiment():
    id: int = field(default_factory=count().__next__, init=False)
    instance_name: str
    model_name: ModelName
    model_parameters: dict
    k_factor: KFactor
    repetitions: int
    worst_solution: Solution = None
    best_solution: Solution = None
    worst_path_length: int = None
    best_path_length: int = None

    def _get_instance(self) -> Instance:
        instance = Instance(
            name=self.instance_name
        )
        instance.get_instance()

        return instance

    def run(self) -> List[Solution]:
        instance = self._get_instance()
        initial_population: List[Solution] = []
        solutions: List[Solution] = []

        if self.model_name == ModelName.ENSEMBLE:
            initial_population_model = NearestNeighbors()
            initial_population = initial_population_model.generate_multiple_solutions(
                instance=instance, 
                k_factor=self.k_factor, 
                n_solutions=self.model_parameters.population_size
            )
            model = GeneticAlgorithm(initial_population, **self.model_parameters)
        
        elif self.model_name == ModelName.GENETIC_ALGORITHM:
            model = GeneticAlgorithm(**self.model_parameters)

        elif self.model_name == ModelName.NEAREST_NEIGHBORS:
            model = NearestNeighbors(**self.model_parameters)

        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        for _ in range(self.repetitions):
            solution = model.generate_solution(
                instance=instance,
                k_factor=self.k_factor
            )

            solutions.append(solution)

        solutions = [i.get_solution_as_dict() for i in solutions]
        
        return solutions