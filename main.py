# %%
from k_tsp_solver import Experiment, ModelName, KFactor, export_results_by_instance, SELECTED_INSTANCES
# %%
genetic_algorithm_parameters = {
    "population_size": 100,
    "generations": 100,
    "mutation_rate": 0.1,
    "selection_size": 10,
    "has_variable_mutate_rate": True
}
nearest_neighbors_parameters = {}
# %%
for instance in SELECTED_INSTANCES: 
    for k_factor in KFactor:
        for has_closed_cycle in [False, True]:
            for model in ModelName:
                experiment = Experiment(
                    instance_name=instance,
                    model_name=model,
                    model_parameters=genetic_algorithm_parameters 
                        if model != ModelName.NEAREST_NEIGHBORS 
                        else nearest_neighbors_parameters,
                    k_factor=k_factor,
                    has_closed_cycle=has_closed_cycle,
                    repetitions=30
                        if model != ModelName.NEAREST_NEIGHBORS 
                        else 1
                )
                experiment.run()
# %%
export_results_by_instance()
# %%
