import uuid
import random
from typing import List
from dataclasses import dataclass, field

from deltalake import write_deltalake
import pandas as pd

from k_tsp_solver import (
    Instance, 
    NearestNeighbors, 
    GeneticAlgorithm, 
    Solution, 
    ModelName, 
    KFactor, 
    dataclass_to_dict,
    logger
)


@dataclass
class ExperimentSession():
    session_id: str = field(default_factory=lambda: str(uuid.uuid1()), init=False)
    results: dict = None
    
@dataclass
class Experiment():
    experiment_name: str
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid1()), init=False)
    instance_name: str
    k_factor: KFactor
    repetitions: int
    model_name: ModelName
    has_closed_cycle: bool = False
    model_parameters: dict = field(default=None, repr=False)
    sessions: List[ExperimentSession] = None
    delta_path: str = field(default="")

    def __post_init__(self):
        self.model_parameters_string = str(self.model_parameters)
        self.delta_path = f"../results/{self.experiment_name}/{self.experiment_id}"

    def _get_instance(self) -> Instance:
        instance = Instance(
            name=self.instance_name
        )
        instance.get_instance()

        return instance

    # def _get_experiment_as_dataframe(self) -> pd.DataFrame:
    #     return (
    #         pd.json_normalize(
    #             data=dataclass_to_dict(self), 
    #             record_path=["sessions"], 
    #             meta=["experiment_id", "model_name"],
    #             sep="_"
    #         )
    #         .rename(columns=lambda x: x.split('results_')[-1])
    #     )

    def _get_experiment_as_dataframe(self) -> pd.DataFrame:
        rows = []
        for session in self.sessions:
            flat = {
                "experiment_id": self.experiment_id,
                "experiment_name": self.experiment_name,
                "instance_name": self.instance_name,
                "k_factor": self.k_factor,
                "repetitions": self.repetitions,
                "model_name": self.model_name,
                "has_closed_cycle": self.has_closed_cycle,
                "session_id": session["session_id"],
                **session["results"],
            }
            rows.append(flat)

        return pd.DataFrame(rows)

    
    def _load_experiment(self) -> None:
        write_deltalake(
            table_or_uri=self.delta_path, 
            data=self._get_experiment_as_dataframe(), 
            mode="append",
            # partition_by=[
            #     "instance_name", 
            #     "k_factor",
            #     "has_closed_cycle",
            #     "model_name"
            # ]
        )

    def run(self):
        instance = self._get_instance()
        initial_population: List[Solution] = []
        sessions: List[dict] = []

        if self.model_name == ModelName.GENETIC_ALGORITHM_NEAREST_NEIGHBORS_ENSEMBLE.value:
            initial_population_model = NearestNeighbors(name=self.model_name)
            initial_population = initial_population_model.generate_multiple_solutions(
                instance=instance, 
                k_factor=self.k_factor,
                has_closed_cycle=self.has_closed_cycle,
                n_solutions=self.model_parameters["population_size"]
            )
            model = GeneticAlgorithm(
                name=self.model_name,
                initial_population=initial_population, 
                **self.model_parameters
            )
        
        elif self.model_name == ModelName.GENETIC_ALGORITHM.value:
            model = GeneticAlgorithm(**self.model_parameters)

        elif self.model_name == ModelName.NEAREST_NEIGHBORS.value:
            model = NearestNeighbors(**self.model_parameters)

        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        logger.info(f"Running experiment for instance {self.instance_name} with model {self.model_name} k-factor {self.k_factor} (closed cycle: {self.has_closed_cycle})")
        for i, _ in enumerate(range(self.repetitions)):
            random.seed(i)
            solution = model.generate_solution(
                instance=instance,
                k_factor=self.k_factor,
                has_closed_cycle=self.has_closed_cycle
            )

            solution_dict = solution.get_solution_as_dict()
            experiment_session = ExperimentSession(results=solution_dict)
            session_dict = dataclass_to_dict(experiment_session)
            sessions.append(session_dict)

        self.sessions = sessions
        self._load_experiment()