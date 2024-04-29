# %%
from k_tsp_solver import Experiment, ModelName, KFactor
import pandas as pd
# %%

experiment = Experiment(
    instance_name="burma14",
    model_name=ModelName.ENSEMBLE,
    model_parameters={},
    k_factor=KFactor.LARGE,
    repetitions=10
)
# %%
solutions = experiment.run()

# %%
pd.json_normalize(solutions)
# %%
