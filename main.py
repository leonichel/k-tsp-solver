# %%
from k_tsp_solver import Experiment, ModelName, KFactor, read_experiments, SELECTED_INSTANCES

# %%
genetic_algorithm_parameters = {
    "population_size": 100,
    "generations": 100,
    "mutation_rate": 0.05,
    "selection_size": 10
}
# %%
for instance in SELECTED_INSTANCES:
    for k_factor in KFactor:
        for model in ModelName:
            experiment = Experiment(
                instance_name="burma14",
                model_name=ModelName.GENETIC_ALGORITHM,
                model_parameters=genetic_algorithm_parameters 
                    if model != ModelName.NEAREST_NEIGHBORS 
                    else {},
                k_factor=KFactor.LARGE,
                repetitions=10
            )
            experiment.run()
# %%
read_experiments()
