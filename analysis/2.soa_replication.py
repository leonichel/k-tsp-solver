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
    "mutation_operator_probabilities": [1/3, 1/3, 1/3, 0.0],
    "use_crossover": False
}
model = ModelName.GENETIC_ALGORITHM_NEAREST_NEIGHBORS_ENSEMBLE.value
# %%
def run_single_experiment(args):
    instance, has_closed, k = args
    experiment = Experiment(
        experiment_name="soa_replication",
        instance_name=instance,
        model_name=model,
        model_parameters=genetic_algorithm_parameters,
        k_factor=k,
        has_closed_cycle=has_closed,
        repetitions=10,
        isolate_delta=True
    )
    experiment.run()
# %%
experiment_args = [
    (instance, has_closed, k)
    for instance in SELECTED_INSTANCES
    for k in KFactor
    for has_closed in [False, True]
]
# %%
Parallel(n_jobs=-1)(
    delayed(run_single_experiment)(args)
    for args in experiment_args
)
# %%
