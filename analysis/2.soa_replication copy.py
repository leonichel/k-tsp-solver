# %%
from joblib import Parallel, delayed
from k_tsp_solver import Experiment, ModelName, KFactor, SELECTED_INSTANCES
# %%
genetic_algorithm_parameters = {
    "population_size": 744,
    "generations": 395,
    "mutation_rate": 0.389328,
    "selection_size": 660,
    "has_variable_mutate_rate": False,
    "mutation_operator_probabilities": [0.11059623110603521, 0.057456092101799985, 0.035713525001528065, 0.7962341517906368],
    "use_crossover": True
}
model = ModelName.GENETIC_ALGORITHM_NEAREST_NEIGHBORS_ENSEMBLE.value
# %%
def run_single_experiment(args):
    instance, has_closed, k = args
    experiment = Experiment(
        experiment_name="final_experiment_with_bayesian_optimization",
        instance_name=instance,
        model_name=model,
        model_parameters=genetic_algorithm_parameters,
        k_factor=k,
        has_closed_cycle=has_closed,
        repetitions=30,
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
