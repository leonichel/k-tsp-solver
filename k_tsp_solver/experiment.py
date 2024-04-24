from k_tsp_solver import Instance, Model, Solution

from itertools import count
from dataclasses import dataclass, field
import math


@dataclass
class Experiment():
    id: int = field(default_factory=count().__next__, init=False)
    instance: Instance
    model: Model
    worst_solution: Solution = None
    best_solution: Solution = None
    worst_path_lenght: int = None
    best_path_lenght: int = None
