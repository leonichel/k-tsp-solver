CREATE OR REPLACE TEMPORARY VIEW results_by_instance AS 
WITH results AS (
    SELECT 
        session_id,
        experiment_id,
        REPLACE(REPLACE(instance_name, '.tsp', ''), '.atsp', '') AS instance_name,
        instance_number_of_vertices AS instance_vertices,
        instance_number_of_edges AS instance_edges,
        instance_symmetrical_type,
        has_closed_cycle,
        k_factor,
        k_size,
        model_name,
        path_length
    FROM
        raw_results
),

results_by_instance AS (
    SELECT 
        instance_name,
        instance_vertices,
        instance_edges,
        instance_symmetrical_type,
        has_closed_cycle,
        k_factor,
        k_size,
        MAX(CASE
            WHEN model_name = 'NearestNeighbors' THEN path_length
            ELSE NULL
        END) AS worst_nearest_neighbors,
        AVG(CASE
            WHEN model_name = 'NearestNeighbors' THEN path_length
            ELSE NULL
        END) AS avg_nearest_neighbors,
        MIN(CASE
            WHEN model_name = 'NearestNeighbors' THEN path_length
            ELSE NULL
        END) AS best_nearest_neighbors,
        MAX(CASE
            WHEN model_name = 'GeneticAlgorithm' THEN path_length
            ELSE NULL
        END) AS worst_genetic_algorithm,
        AVG(CASE
            WHEN model_name = 'GeneticAlgorithm' THEN path_length
            ELSE NULL
        END) AS avg_genetic_algorithm,
        MIN(CASE
            WHEN model_name = 'GeneticAlgorithm' THEN path_length
            ELSE NULL
        END) AS best_genetic_algorithm,
        MAX(CASE
            WHEN model_name = 'Ensemble' THEN path_length
            ELSE NULL
        END) AS worst_ensemble,
        AVG(CASE
            WHEN model_name = 'Ensemble' THEN path_length
            ELSE NULL
        END) AS avg_ensemble,
        MIN(CASE
            WHEN model_name = 'Ensemble' THEN path_length
            ELSE NULL
        END) AS best_ensemble,
        MAX(path_length) AS worst_solution,
        AVG(path_length) AS avg_solution,
        MIN(path_length) AS best_solution
    FROM
        results
    GROUP BY
        instance_name,
        instance_vertices,
        instance_edges,
        instance_symmetrical_type,
        has_closed_cycle,
        k_factor,
        k_size
    ORDER BY
        instance_name,
        instance_symmetrical_type,
        has_closed_cycle,
        k_factor
)

SELECT *
FROM results_by_instance;

COPY results_by_instance TO 'experiments/summary_results.csv';