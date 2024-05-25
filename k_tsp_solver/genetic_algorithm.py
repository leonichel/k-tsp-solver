import random
from queue import Queue
from typing import List
from dataclasses import dataclass, field

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
class GeneticAlgorithm(Model):
    name: str = ModelName.GENETIC_ALGORITHM
    population_size: int = 100
    generations: int = 100
    mutation_rate: float = 0.05
    selection_size: int = 10
    diversity_rate: float = field(default=None, repr=False)
    best_solution: Solution = field(default=None, repr=False)
    best_path_length: int = field(default=None, repr=False)
    initial_population: List[Solution] = field(default=None, repr=False)
    has_variable_mutate_rate: bool = field(default=True, repr=False)
    is_debugging: bool = field(default=False, repr=False)

    def get_edge_to_close_cycle(self, instance: Instance, path_edges: list) -> tuple:
        first_vertex = path_edges[0][0]
        last_vertex = path_edges[-1][1]

        return last_vertex, first_vertex, instance.graph[last_vertex][first_vertex]

    def generate_random_solution(self, instance: Instance, k_factor: KFactor, has_closed_cycle: bool) -> list:
        path_edges = []
        visited = set()
        last_vertex = None

        solution = Solution(
            instance=instance,
            model=self,
            k_factor=k_factor,
            has_closed_cycle=has_closed_cycle,
            path_edges=path_edges
        )

        vertices_to_append = solution.k_size - 2 if has_closed_cycle else solution.k_size - 1

        for _ in range(vertices_to_append):
            random_edge = random.choice([
                (u, v, d) 
                for u, v, d in instance.graph.edges(nbunch=last_vertex, data=True)
                if u != v and v not in visited
            ])
            path_edges.append(random_edge)
            visited.add(random_edge[0])
            visited.add(random_edge[1])
            last_vertex = random_edge[1]

            if has_closed_cycle:
                edge_to_close_cycle = self.get_edge_to_close_cycle(instance=instance, path_edges=path_edges)
                path_edges.append(edge_to_close_cycle)

        solution.evaluate_edge_path_length()
        solution.get_path_vertices()

        return solution
    
    def generate_random_population(self, instance: Instance, k_factor: KFactor, has_closed_cycle:bool) -> List[Solution]:
        population: List[Solution] = []

        for _ in range(self.population_size):
            solution = self.generate_random_solution(
                instance=instance, 
                k_factor=k_factor,
                has_closed_cycle=has_closed_cycle
            )
            population.append(solution)

        return population
    
    def roulette_selection(self, population: List[Solution]) -> List[Solution]:
        fitness = [
            (1/solution.path_length)
            if solution.path_length != 0 
            else 0
            for solution in population 
        ]
        total_fit = sum(fitness)
        relative_fit = [f/total_fit for f in fitness]

        return random.choices(population, weights=relative_fit, k=self.selection_size)
    
    def mutate(self, solution: Solution) -> Solution:
        def swap_mutation(solution: Solution) -> Solution:
            i = random.randint(0, len(solution.path_vertices) - 1)
            j = random.randint(0, len(solution.path_vertices) - 1)
            solution.path_vertices[i], solution.path_vertices[j] = solution.path_vertices[j], solution.path_vertices[i]

            return solution
        
        def reverse_swap_mutation(solution: Solution) -> Solution:
            i = random.randint(0, len(solution.path_vertices) - 1)
            j = random.randint(0, len(solution.path_vertices) - 1)
            start = min(i, j)
            end = max(i, j)
            solution.path_vertices[start:end+1] = solution.path_vertices[start:end+1][::-1]

            return solution
            
        def slide_mutation(solution: Solution) -> Solution:
            slide_point_value = random.choice(solution.path_vertices)
            slide_index = random.randint(0, len(solution.path_vertices) - 1)
            solution.path_vertices.remove(slide_point_value)
            solution.path_vertices.insert(slide_index, slide_point_value)

            return solution
            
        def replacement_mutation(solution: Solution) -> Solution:
            graph = solution.instance.graph
            i = random.randint(0, len(solution.path_vertices) - 1)

            if i == 0:
                next_vertex = solution.path_vertices[i + 1]
                replacement_edges = sorted(
                    [
                        (u, v, d) 
                        for u, v, d in graph.edges(data=True) 
                        if u != v 
                            and u not in solution.path_vertices
                            and v == next_vertex
                    ], 
                    key=lambda x: x[2]['weight']
                )
                replacement_vertices = [u for (u, _, _) in replacement_edges]

            elif i == (len(solution.path_vertices) - 1):
                previous_vertex = solution.path_vertices[i - 1]
                replacement_edges = sorted(
                    [
                        (u, v, d) 
                        for u, v, d in graph.edges(data=True) 
                        if u != v 
                            and v not in solution.path_vertices
                            and u == previous_vertex
                    ], 
                    key=lambda x: x[2]['weight']
                )
                replacement_vertices = [v for (_, v, _) in replacement_edges]

            else:
                previous_city = solution.path_vertices[i - 1]
                next_vertex = solution.path_vertices[i + 1]
                candidate_nodes = [v for v in graph.nodes() if v not in solution.path_vertices]
                weights = {
                    n: (graph[previous_city][n]['weight'], graph[n][next_vertex]['weight']) 
                    for n in candidate_nodes
                }
                replacement_vertices = sorted(
                    weights.items(), key=lambda item: sum(item[1]), reverse=False
                )
                replacement_vertices = [u[0] for u in replacement_vertices]

            if len(replacement_vertices) != 0:
                nearest_intermediate_vertex = replacement_vertices[0]
                solution.path_vertices[i] = nearest_intermediate_vertex

            return solution
        
        mutate_function = random.choice([
            swap_mutation, 
            reverse_swap_mutation, 
            slide_mutation, 
            replacement_mutation
        ])

        mutated_solution = mutate_function(solution)

        return mutated_solution
    
    def ordered_crossover(self, solution_1: Solution, solution_2: Solution) -> Solution:
        solution_1_path = solution_1.path_vertices
        solution_2_path = solution_2.path_vertices

        if solution_1.has_closed_cycle:
            solution_1_path = solution_1_path[:-1]
            solution_2_path = solution_2_path[:-1]

        crossover_point_1 = random.randint(0, len(solution_1_path) - 1)
        crossover_point_2 = random.randint(0, len(solution_1_path) - 1)

        if crossover_point_2 < crossover_point_1:
            crossover_point_1, crossover_point_2 = crossover_point_2, crossover_point_1

        offspring = solution_1.path_vertices[crossover_point_1:crossover_point_2]
        index = crossover_point_2

        while len(offspring) < len(solution_1_path):
            edge = solution_2.path_vertices[index % len(solution_2.path_vertices)]

            if edge not in offspring:
                offspring.append(edge)

            index += 1

        solution = Solution(
            instance=solution_1.instance,
            model=solution_1.model,
            k_factor=solution_1.k_factor,
            has_closed_cycle=solution_1.has_closed_cycle,
            path_vertices=offspring
        )

        if random.random() <= self.mutation_rate:
            solution = self.mutate(solution)

        if solution.has_closed_cycle:
            solution.path_vertices.append(solution.path_vertices[0])

        solution.get_path_edges()
        solution.evaluate_edge_path_length()

        return solution
    
    def generate_offspring_population(self, population: List[Solution]) -> List[Solution]:
        next_generation: List[Solution] = []
        number_of_offsprings = self.population_size - len(population)

        for _ in range(number_of_offsprings):
            solution_1 = random.choice(population)
            solution_2 = random.choice(population)
            offspring = self.ordered_crossover(solution_1, solution_2)
            next_generation.append(offspring)

        offspring_population = population + next_generation
        
        return offspring_population
    
    def set_variable_mutate_rate(self) -> None:
        mutate_rate = 1 - self.diversity_rate

        if mutate_rate < 0.05:
            mutate_rate = 0.05
        elif mutate_rate > 0.75:
            mutate_rate = 0.75
        
        self.mutation_rate = mutate_rate

    def evaluate_population(self, population: List[Solution]) -> tuple:
        best_solution, best_path_length = sorted(
            [
                (solution, solution.path_length)
                for solution in population
            ],
            key=lambda x: x[1]
        )[0]
        size = len(population)
        distinct_solutions = len(set([
            tuple(solution.path_vertices)
            for solution in population
        ]))
        diversity_rate = round(distinct_solutions / size, 4)

        if self.best_path_length is None: 
            self.best_path_length = float("inf")

        if best_path_length < self.best_path_length:
            self.best_solution = best_solution
            self.best_path_length = best_path_length

        self.diversity_rate = diversity_rate

        if self.has_variable_mutate_rate:
            self.set_variable_mutate_rate()

        return best_solution, best_path_length, diversity_rate

    @timeit
    def generate_solution(self, instance: Instance, k_factor: KFactor, has_closed_cycle: bool):
        self.diversity_rate: float = None
        self.best_solution: Solution = None
        self.best_path_length: int = None

        if self.initial_population is not None:
            population = self.initial_population
        else:
            population = self.generate_random_population(
                instance=instance, 
                k_factor=k_factor,
                has_closed_cycle=has_closed_cycle
            )

        for i in range(self.generations):
            _, best_path_length, diversity_rate = self.evaluate_population(population)

            if self.is_debugging:
                logger.info("Generation %d: Best path length = %d, Diversity rate = %.2f", i, best_path_length, diversity_rate)
                
            selected_population = self.roulette_selection(population)
            population = self.generate_offspring_population(selected_population)

        return self.best_solution