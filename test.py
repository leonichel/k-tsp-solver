#%%
from k_tsp_solver import Instance, Experiment, ModelName, KFactor, export_results_by_instance, SELECTED_INSTANCES
import numpy as np
import random

random.seed(0)
# %%
instance = Instance(name="att48.tsp")
instance.get_instance()
# %%
initial_pheromone = 1.0
beta = 0.3
alpha = 0.7
# %%
pheromone = np.full((instance.number_of_vertices, instance.number_of_vertices), initial_pheromone)
# %%
# generate solutions
## for loop for each ant
### call generate_solution
path = []
visited = set()
first_edge = random.randint(1, instance.number_of_vertices)
visited.add(first_edge)
current = first_edge
path.append(current)
# %%
path
# %%
# construct path loop
## _select_next_node
### _calculate_probabilities

def _calculate_probability(instance: Instance, current: int, next_node: int, pheromone: np.ndarray) -> float:
    pheromone_factor = pheromone[current - 1, next_node - 1] ** alpha
    distance_factor = 1 / instance.graph.get_edge_data(current, next_node)["weight"] ** beta
    return pheromone_factor * distance_factor

def _calculate_probabilities(instance: Instance, current: int, visited: set, pheromone: np.ndarray) -> np.ndarray:
        unvisited = instance.graph.nodes - visited
        probabilities = np.zeros(instance.number_of_vertices)
        for node in unvisited:
            probabilities[node - 1] = _calculate_probability(instance, current, node, pheromone)
        return probabilities / np.sum(probabilities)

probabilities = _calculate_probabilities(instance, current, visited, pheromone)
# %%
# select next node
np.random.choice(instance.number_of_vertices, p=probabilities)
# %%
