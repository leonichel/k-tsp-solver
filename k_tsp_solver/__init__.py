import logging


logger = logging.getLogger("k_tsp_solver")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s"
)

handler.setFormatter(formatter)
logger.addHandler(handler)


from .utils import timeit, dataclass_to_dict, read_experiments
from .instance import Instance
from .model import Model
from .enums import ModelName, KFactor
from .solution import Solution
from .nearest_neighbors import NearestNeighbors
from .genetic_algorithm import GeneticAlgorithm
from .experiment import Experiment
from .constants import SELECTED_INSTANCES


__all__ = [
    "timeit",
    "dataclass_to_dict",
    "read_experiments",
    "Instance", 
    "Model",
    "ModelName",
    "KFactor",
    "Solution",
    "NearestNeighbors",
    "GeneticAlgorithm",
    "Experiment",
    "SELECTED_INSTANCES"
]