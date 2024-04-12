# %%
from k_tsp_solver import Instance, GeneticAlgorithm, NearestNeighbors, Solution

from typing import List

# %%
instance = Instance(
    name="burma14" # rat783
)
instance.get_instance()

instance
# %%
genetic_algorithm = GeneticAlgorithm()

# %%

import time

start_time = time.time()

population = genetic_algorithm.generate_random_population(
    instance=instance, 
    k_factor=3/4
)

end_time = time.time()

end_time - start_time

# %%
selected_population = genetic_algorithm.roulette_selection(population)

# %%
population = genetic_algorithm.generate_next_generation(selected_population)
# %%
solution = population[0]
solution.get_path_vertices()
solution.path_vertices

# %%
# genetic_algorithm.mutate(population[0])
# %%
import random

i = random.randint(0, len(solution.path_vertices) - 1)
j = random.randint(0, len(solution.path_vertices) - 1)
solution.path_vertices[i], solution.path_vertices[j] = solution.path_vertices[j], solution.path_vertices[i]
# %%
solution.path_vertices
# %%
