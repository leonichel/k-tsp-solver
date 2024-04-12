from k_tsp_solver import Instance

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Model(ABC):
    name: str

    @abstractmethod
    def generate_solution(self, instance: Instance):
        pass 
