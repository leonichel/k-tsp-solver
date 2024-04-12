from k_tsp_solver import Instance, Solution, Model

from dataclasses import dataclass
from typing import List
import random


@dataclass
class GeneticAlgorithm(Model):
    name: str = "GeneticAlgorith"
    population_size: int = 100
    generations: int = 100
    mutation_rate: float = 0.05
    selection_size: int = 10


    def generate_random_solution(self, instance: Instance, k_factor: float) -> list:
        path_edges = []
        visited = set()
        last_vertex = None

        solution = Solution(
            instance=instance,
            model=self.name,
            k_factor=k_factor,
            path_edges=path_edges
        )

        for _ in range(solution.k_size - 1):
            random_edge = random.choice([
                (u, v, d) 
                for u, v, d in instance.graph.edges(nbunch=last_vertex, data=True)
                if u != v and v not in visited
            ])
            path_edges.append(random_edge)
            visited.add(random_edge[0])
            visited.add(random_edge[1])
            last_vertex = random_edge[1]

        solution.evaluate_edge_path_lenght()

        return solution
    
    def generate_random_population(self, instance: Instance, k_factor: float) -> List[Solution]:
        population: List[Solution] = []

        for _ in range(self.population_size):
            solution = self.generate_random_solution(
                instance=instance, 
                k_factor=k_factor
            )
            population.append(solution)

        return population
    
    def roulette_selection(self, population: List[Solution]) -> List[Solution]:
        fitness = [solution.path_length for solution in population]
        total_fit = sum(fitness)
        relative_fit = [f/total_fit for f in fitness]

        return random.choices(population, weights=relative_fit, k=self.selection_size)
    
    def ordered_crossover(self, solution_1: Solution, solution_2: Solution) -> Solution:
        solution_1.get_path_vertices()
        solution_2.get_path_vertices()

        crossover_point_1 = random.randint(0, len(solution_1.path_vertices) - 1)
        crossover_point_2 = random.randint(0, len(solution_1.path_vertices) - 1)

        if crossover_point_2 < crossover_point_1:
            crossover_point_1, crossover_point_2 = crossover_point_2, crossover_point_1

        offspring = solution_1.path_vertices[crossover_point_1:crossover_point_2]
        index = crossover_point_2

        while len(offspring) < len(solution_1.path_vertices):
            edge = solution_2.path_vertices[index % len(solution_2.path_vertices)]

            if edge not in offspring:
                offspring.append(edge)

            index += 1

        solution = Solution(
            instance=solution_1.instance,
            model=solution_1.model,
            k_factor=solution_1.k_factor,
            path_vertices=offspring
        )

        solution.get_path_edges()
        solution.evaluate_edge_path_lenght()

        return solution
    
    def generate_next_generation(self, population: List[Solution]) -> List[Solution]:
        next_generation: List[Solution] = []

        while len(next_generation) < self.population_size:
            solution_1 = random.choice(population)
            solution_2 = random.choice(population)
            solution = self.ordered_crossover(solution_1, solution_2)
            next_generation.append(solution)
        
        return next_generation
    
    def mutate(self, solution: Solution) -> Solution:
        def swap_mutation(solution: Solution) -> Solution:
            mutated_solution = solution.copy()

            for i in range(len(mutated_solution)):
                j = random.randint(0, len(mutated_solution) - 1)
                mutated_solution[i], mutated_solution[j] = mutated_solution[j], mutated_solution[i]

            return mutated_solution
        
        def reverse_swap_mutation(solution: Solution) -> Solution:
            mutated_solution = solution.copy()

            for i in range(len(mutated_solution)):
                j = random.randint(0, len(mutated_solution) - 1)
                start = min(i, j)
                end = max(i, j)
                mutated_solution[start:end+1] = mutated_solution[start:end+1][::-1]

            return mutated_solution
            
        def slide_mutation(solution: Solution) -> Solution:
            slide_point_value = random.choice(solution.path_vertices)
            slide_index = random.randint(0, len(solution.path_vertices) - 1)
            solution.path_vertices.remove(slide_point_value)
            solution.path_vertices.insert(slide_index, slide_point_value)
            solution.get_path_edges()
            solution.evaluate_edge_path_lenght()

            return solution
            
        def replacement_mutation(solution: Solution) -> Solution:
            return None
        
        mutate_function = random.choice([
            swap_mutation, 
            # reverse_swap_mutation, 
            # slide_mutation, 
            # replacement_mutation
        ])

        solution.get_path_vertices()
        mutated_solution = mutate_function(solution)
        mutated_solution.get_path_edges()
        mutated_solution.evaluate_edge_path_lenght()

        return mutated_solution
    
    def generate_solution(self, instance: Instance):
        return None

    # def generate_solution(df: pd.DataFrame, cities: np.array, solution_size: int, index: int) -> np.array:
    #     solution = np.zeros(solution_size, dtype=int)
    #     not_available_cities = set()

    #     if index >= solution_size:
    #         index = int(np.ceil(index / 2))

    #     first_city = get_first_city_of_solution(df, index)
    #     solution[0] = first_city
    #     not_available_cities.add(first_city)

    #     for i in range(1, solution_size):
    #         city = solution[i - 1]
    #         neighbors = get_neighbors(df, city, not_available_cities)
    #         nearest_city = get_nearest_city(neighbors)

    #         if nearest_city is None:
    #         break

    #         solution[i] = nearest_city
    #         not_available_cities.add(nearest_city)

    #     return solution

    # def generate_initial_population(df: pd.DataFrame, population_size: int, k_factor: float) -> np.array:
    #     cities = df['source'].unique()
    #     cities_number = cities.size
    #     solution_size = get_solution_size(cities_number, k_factor)
    #     population = np.zeros((population_size, solution_size), dtype=int)
    #     solution_index = 0

    #     for population_index in range(population_size):
    #         is_solution_feasible = False

    #         while not is_solution_feasible:
    #         solution = generate_solution(df, cities, solution_size, solution_index)
    #         is_solution_feasible = check_if_solution_is_feasible(df, solution, solution_size)

    #         if is_solution_feasible:
    #             population[population_index] = solution

    #         solution_index += 1

    #     return population

    # def generate_random_initial_population(df, population_size, k_factor):
    #     population = []
    #     cities = list(df['source'].unique())
    #     num_cities = k_factor(len(cities))

    #     for cromossome in range(population_size):
    #         is_solution_feasible = False
    #         while not is_solution_feasible:
    #         solution = np.random.choice(np.arange(1, len(cities) + 1), size=num_cities, replace=False).tolist()
    #         is_solution_feasible = check_if_solution_is_feasible(df, solution, k_factor)
    #         if is_solution_feasible:
    #             population.append(solution)

    #     return population

    # def roulette_selection(population: np.array, fitness_values: np.array, num_select: int) -> np.array:
    #     selected_indices = np.zeros(num_select, dtype=int)
    #     fitness_sum = np.sum(fitness_values)
    #     probabilities = (fitness_sum - fitness_values) / fitness_sum
    #     probabilities /= np.sum(probabilities)
    #     electist_selection_index = np.argmin(fitness_values)
    #     selected_indices[0] = electist_selection_index
    #     selected_indices[1:] = np.random.choice(len(population), size=num_select - 1, p=probabilities, replace=False)
    #     selected = population[selected_indices]
    #     selected_fitness_values = fitness_values[selected_indices]
    #     sorted_indices = np.argsort(selected_fitness_values)
    #     sorted_population = selected[sorted_indices]

    #     return sorted_population
    
    # def swap_mutation(solution: np.array, mutation_rate: float) -> np.array:
    # mutated_solution = solution.copy()
    # for i in range(len(mutated_solution)):
    #     if np.random.random() < mutation_rate:
    #     j = np.random.randint(0, len(mutated_solution) - 1)
    #     mutated_solution[i], mutated_solution[j] = mutated_solution[j], mutated_solution[i]

    # return mutated_solution

    # def reverse_swap_mutation(solution: np.array, mutation_rate: float) -> np.array:
    # mutated_solution = solution.copy()
    # for i in range(len(mutated_solution)):
    #     if np.random.random() < mutation_rate:
    #     j = np.random.randint(0, len(mutated_solution) - 1)
    #     start = min(i, j)
    #     end = max(i, j)
    #     mutated_solution[start:end+1] = np.flip(mutated_solution[start:end+1])

    # return mutated_solution

    # def slide_mutation(solution: np.array, mutation_rate: float) -> np.array:
    # mutated_solution = solution.copy()
    # for i in range(len(mutated_solution)):
    #     if np.random.random() < mutation_rate:
    #     slide_index = np.random.randint(0, len(mutated_solution) - 1)
    #     slide_value = mutated_solution[i]
    #     mutated_solution = np.delete(mutated_solution, i)
    #     mutated_solution = np.insert(mutated_solution, slide_index, slide_value)

    # return mutated_solution

    # def add_new_city_mutation(solution: np.array, mutation_rate: float) -> np.array:
    # for i in range(len(solution)):
    #     if np.random.random() < mutation_rate:
    #     if i == 0:
    #         next_city = solution[i+1]
    #         closest_intermediate_city = df.query('target == @next_city and source not in @solution').sort_values('weight')
    #     if i == (len(solution) - 1):
    #         before_city = solution[i-1]
    #         closest_intermediate_city = df.query('source == @before_city and target not in @solution').sort_values('weight')
    #     else:
    #         before_city = solution[i-1]
    #         next_city = solution[i+1]
    #         closest_intermediate_city = df.query('(target == @before_city and source not in @solution) or (target == @next_city and source not in @solution)').groupby(['source']).sum().sort_values('weight')

    #     if closest_intermediate_city.shape[0] != 0:
    #         solution[i] = closest_intermediate_city.iloc[0].name

    # return solution

    # def mutate(solution: np.array, mutation_rate: float) -> np.array:
    # mutate_functions = [swap_mutation, reverse_swap_mutation, slide_mutation, add_new_city_mutation]
    # selected_functions = np.random.choice(mutate_functions, p=mutate_functions_weights)
    # mutated_solution = selected_functions(solution, mutation_rate)

    # return mutated_solution

    # def mono_crossover(*args: np.array) -> np.array:
    # return args[0]

    # def pairwise_crossover(*args: np.array) -> np.array:
    # min_length = len(args[0])
    # crossover_point = np.random.randint(1, min_length)
    # child = np.concatenate((args[0][:crossover_point], args[1][crossover_point:]))

    # return child

    # def ternary_crossover(*args: np.array) -> np.array:
    # child = []

    # for i in range(len(args[0])):
    #     if np.random.random() < 0.5:
    #     child.append(args[0][i])
    #     else:
    #     if np.random.random() < 0.5:
    #         child.append(args[1][i])
    #     else:
    #         child.append(args[2][i])

    # return np.array(child)