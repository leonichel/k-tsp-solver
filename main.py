# %%
from k_tsp_solver import Instance, GeneticAlgorithm, NearestNeighbors

# %%
instance = Instance(
    name="burma14" # rat783, burma14
)
instance.get_instance()

instance
# %%
genetic_algorithm = GeneticAlgorithm(
    population_size=100,
    generations=100,
    mutation_rate=0.05,
    selection_size=10
)

# %%

solution = genetic_algorithm.generate_solution(instance=instance, k_factor=3/4, is_debugging=False)
# %%

solution, solution.path_length
# %%
