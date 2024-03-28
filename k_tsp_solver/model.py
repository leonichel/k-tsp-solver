from k_tsp_solver import Instance

from abc import ABC, abstractmethod
from dataclasses import dataclass

from graphframes import GraphFrame
from pyspark.sql import Row

from pyspark.sql import functions as F


@dataclass
class Model(ABC):
    name: str
    parameters: dict

    @abstractmethod
    def generate_solution(self, instance: Instance):
        pass 
