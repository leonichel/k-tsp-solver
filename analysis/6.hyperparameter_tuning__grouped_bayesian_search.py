# %%
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import optuna

from k_tsp_solver import Experiment, ModelName, KFactor
# %%
SMALL_TSP_INSTANCES = ["gr24.tsp", "berlin52.tsp", "pr76.tsp"]
MEDIUM_TSP_INSTANCES = ["ch130.tsp", "pr152.tsp", "kroA200.tsp"]
LARGE_TSP_INSTANCES = ["att532.tsp", "si535.tsp", "rat783.tsp"]
TSP_SELECTED_INSTANCES = SMALL_TSP_INSTANCES + MEDIUM_TSP_INSTANCES + LARGE_TSP_INSTANCES

ATSP_SELECTED_INSTANCES = ["br17.atsp", "ftv170.atsp", "rbg403.atsp"]

SELECTED_INSTANCES = TSP_SELECTED_INSTANCES + ATSP_SELECTED_INSTANCES

INSTANCE_GROUPS = {
    "small": SMALL_TSP_INSTANCES,
    "medium": MEDIUM_TSP_INSTANCES,
    "large": LARGE_TSP_INSTANCES,
    "atsp": ATSP_SELECTED_INSTANCES
}

MODEL = ModelName.GENETIC_ALGORITHM_NEAREST_NEIGHBORS_ENSEMBLE.value
# %%
def run_single_experiment(args):
    instance, has_closed, k, params_dict = args
    experiment = Experiment(
        experiment_name="hyperparameter_tuning__grouped_bayesian_search",
        instance_name=instance,
        model_name=MODEL,
        model_parameters=params_dict,
        k_factor=k,
        has_closed_cycle=has_closed,
        repetitions=1,
        isolate_delta=True,
    )
    experiment.run()
    return experiment._get_experiment_as_dataframe()["path_length"].min()

def make_objective(instances, fixed_k=None):
    def objective(trial):
        rng = np.random.default_rng(seed=trial.number)
        mutation_operator_probabilities = rng.dirichlet([1, 1, 1, 1])
        trial.set_user_attr("mutation_operator_probabilities", mutation_operator_probabilities.tolist())
        generations = trial.suggest_int("generations", 10, 1000)
        population_size = trial.suggest_int("population_size", 10, 1000)
        selection_size = trial.suggest_int("selection_size", 1, population_size - 1)
        mutation_rate = trial.suggest_float("mutation_rate", 0.001, 0.999)
        has_variable_mutate_rate = trial.suggest_categorical("has_variable_mutate_rate", [True, False])
        use_crossover = trial.suggest_categorical("use_crossover", [True, False])

        params_dict = {
            "population_size": population_size,
            "generations": generations,
            "mutation_rate": mutation_rate,
            "selection_size": selection_size,
            "has_variable_mutate_rate": has_variable_mutate_rate,
            "use_crossover": use_crossover,
            "mutation_operator_probabilities": mutation_operator_probabilities,
            "is_debugging": False
        }

        args = []
        for inst in instances:
            for k in ( [fixed_k] if fixed_k is not None else list(KFactor) ):
                for closed in (False, True):
                    args.append((inst, closed, k, params_dict))

        results = Parallel(n_jobs=-1)(delayed(run_single_experiment)(a) for a in args)
        return float(np.mean(results))
    
    return objective
# %%
for k in KFactor:
    print(f"Running hyperparameter tuning for k={k.name}...")
    objective = make_objective(SELECTED_INSTANCES, fixed_k=k)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=0)
    )
    study.optimize(objective, n_trials=100)
    
    print("Best parameters found:")
    print(study.best_params)
    print("Best average objective value:")
    print(study.best_value)
    
    df = study.trials_dataframe()
    df.to_csv(f"hyperparameter_tuning__bayesian_search__results_k_{k.name}.csv", index=False)
# %%
for group_name, instances in INSTANCE_GROUPS.items():
    print(f"Running hyperparameter tuning for group '{group_name}'...")
    objective = make_objective(instances)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=0)
    )
    study.optimize(objective, n_trials=100)
    print("Best parameters found:")
    print(study.best_params)
    print("Best average objective value:")
    print(study.best_value)

    df = study.trials_dataframe()
    df.to_csv(f"hyperparameter_tuning__bayesian_search__results_group_{group_name}.csv", index=False)