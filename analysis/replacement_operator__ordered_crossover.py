# %%
from joblib import Parallel, delayed
from k_tsp_solver import Experiment, ModelName, KFactor, SELECTED_INSTANCES
# %%
genetic_algorithm_parameters = {
    "population_size": 100,
    "generations": 100,
    "mutation_rate": 0.01,
    "selection_size": 10,
    "has_variable_mutate_rate": False,
    "mutation_operator_probabilities": [1/4, 1/4, 1/4, 1/4],    # change
    "use_crossover": True   # change
}
nearest_neighbors_parameters: dict = {}
params_dict = lambda model: (
    nearest_neighbors_parameters 
    if model == ModelName.NEAREST_NEIGHBORS 
    else genetic_algorithm_parameters
)
# %%
def run_single_experiment(args):
    instance, model, has_closed, k, params_dict = args
    experiment = Experiment(
        experiment_name="replacement_operator__ordered_crossover",
        instance_name=instance,
        model_name=model.value,
        model_parameters=params_dict,
        k_factor=k,
        has_closed_cycle=has_closed,
        repetitions=10
    )
    experiment.run()
# %%
experiment_args = [
    (instance, model, has_closed, k, params_dict(model))
    for instance in SELECTED_INSTANCES
    for model in ModelName
    for k in KFactor
    for has_closed in [False, True]
]
# %%
Parallel(n_jobs=-1)(
    delayed(run_single_experiment)(args)
    for args in experiment_args
)
# %%
