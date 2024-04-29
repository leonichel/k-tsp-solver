from enum import Enum


class ModelName(Enum):
    NEAREST_NEIGHBORS = "NearestNeighbors"
    GENETIC_ALGORITHM = "GeneticAlgorithm"
    ENSEMBLE = "Ensemble"


class KFactor(Enum):
    SMALL = 1/4
    MEDIUM = 1/2
    LARGE = 3/4