from enum import Enum


class ModelName(Enum):
    NEAREST_NEIGHBORS = "NearestNeighbors"
    NEAREST_NEIGHBORS_V2 = "NearestNeighborsV2"
    GENETIC_ALGORITHM = "GeneticAlgorithm"
    GENETIC_ALGORITHM_NEAREST_NEIGHBORS_ENSEMBLE = "GeneticAlgorithmNearestNeighborsEnsemble"

class KFactor(Enum):
    SMALL = 1/4
    MEDIUM = 1/2
    LARGE = 3/4