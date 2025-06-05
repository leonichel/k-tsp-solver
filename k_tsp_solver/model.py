from abc import ABC, abstractmethod
from dataclasses import dataclass

from k_tsp_solver import Instance


@dataclass
class Model(ABC):
    name: str
