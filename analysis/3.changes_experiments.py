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

replace_operator_parameters = genetic_algorithm_parameters.copy()
replace_operator_parameters["mutation_operator_probabilities"] = [1/4, 1/4, 1/4, 1/4]

variable_mutate_rate_parameters = genetic_algorithm_parameters.copy()
variable_mutate_rate_parameters["has_variable_mutate_rate"] = True

ordered_crossover_parameters = genetic_algorithm_parameters.copy()
ordered_crossover_parameters["use_crossover"] = True

replacement_operator__ordered_crossover_parameters = genetic_algorithm_parameters.copy()
replacement_operator__ordered_crossover_parameters["mutation_operator_probabilities"] = [1/4, 1/4, 1/4, 1/4]
replacement_operator__ordered_crossover_parameters["use_crossover"] = True

variable_mutate_rate__ordered_crossover_parameters = genetic_algorithm_parameters.copy()
variable_mutate_rate__ordered_crossover_parameters["has_variable_mutate_rate"] = True
variable_mutate_rate__ordered_crossover_parameters["use_crossover"] = True

variable_mutate_rate__replacement_operator_parameters = genetic_algorithm_parameters.copy()
variable_mutate_rate__replacement_operator_parameters["has_variable_mutate_rate"] = True
variable_mutate_rate__replacement_operator_parameters["mutation_operator_probabilities"] = [1/4, 1/4, 1/4, 1/4]

all_changes_parameters = genetic_algorithm_parameters.copy()
all_changes_parameters["has_variable_mutate_rate"] = True
all_changes_parameters["use_crossover"] = True
all_changes_parameters["mutation_operator_probabilities"] = [1/4, 1/4, 1/4, 1/4]

parameters = {
    "soa_replication": genetic_algorithm_parameters,
    "replace_operator": replace_operator_parameters,
    "variable_mutate_rate": variable_mutate_rate_parameters,
    "ordered_crossover": ordered_crossover_parameters,
    "replace_operator__ordered_crossover": replacement_operator__ordered_crossover_parameters,
    "variable_mutate_rate__ordered_crossover": variable_mutate_rate__ordered_crossover_parameters,
    "variable_mutate_rate__replace_operator": variable_mutate_rate__replacement_operator_parameters,
    "all_changes": all_changes_parameters
}
model = ModelName.GENETIC_ALGORITHM_NEAREST_NEIGHBORS_ENSEMBLE.value
# %%
def run_single_experiment(args):
    instance, has_closed, k, experiment_name = args
    experiment = Experiment(
        experiment_name=f"change__{experiment_name}",
        instance_name=instance,
        model_name=model,
        model_parameters=parameters[experiment_name],
        k_factor=k,
        has_closed_cycle=has_closed,
        repetitions=30,
        isolate_delta=True
    )
    experiment.run()
# %%
for experiment_name in parameters.keys():
    print(f"Running experiment: {experiment_name}")
    experiment_args = [
        (instance, has_closed, k, experiment_name)
        for instance in SELECTED_INSTANCES
        for k in KFactor
        for has_closed in [False, True]
    ]

    Parallel(n_jobs=-1)(
        delayed(run_single_experiment)(args)
        for args in experiment_args
    )
# %%
