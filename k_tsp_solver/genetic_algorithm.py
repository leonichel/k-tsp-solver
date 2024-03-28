from k_tsp_solver import Instance, Solution, Model

from dataclasses import dataclass
from pyspark.sql import functions as F

from graphframes import GraphFrame
from pyspark.sql import Row


@dataclass
class GeneticAlgorithm(Model):
    population_size: int
    generations: int = 100
    mutation_rate: float = 0.05
    selection_size: int = 10

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