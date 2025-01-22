from enum import Enum


class ModelName(Enum):
    NEAREST_NEIGHBORS = "NearestNeighbors"
    GENETIC_ALGORITHM = "GeneticAlgorithm"
    SWARM_OPTIMIZATION_ALGORITHM = "SwarmOptimizationAlgorithm"
    GENETIC_ALGORITHM_NEAREST_NEIGHBORS_ENSEMBLE = "GeneticAlgorithmNearestNeighborsEnsemble"
    ANT_COLONY_OPTIMIZATION = "AntColonyOptimization"
    ANT_COLONY_OPTIMIZATION_ENSEMBLE = "AntColonyOptimizationEnsemble"

class KFactor(Enum):
    SMALL = 1/4
    MEDIUM = 1/2
    LARGE = 3/4