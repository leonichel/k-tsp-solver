import os
import pandas as pd
import pyarrow as pa
from typing import List
from joblib import Parallel, delayed

from deltalake import write_deltalake, DeltaTable
from k_tsp_solver import logger, Solution, Instance, NearestNeighborsV2, KFactor, SELECTED_INSTANCES 
from k_tsp_solver.utils import dataclass_to_dict


# def initialize_delta_table(delta_path: str):
#     if not os.path.exists(delta_path):
#         logger.info("Initializing Delta table structure...")
#         schema = pa.schema([
#             ("instance_name", pa.string()),
#             ("k_factor", pa.string()),
#             ("has_closed_cycle_cycle", pa.bool_()),
#             ("path_vertices", pa.list_(pa.int32())),
#             ("path_length", pa.int32())
#         ])

#         # Create a dummy table with one row to initialize schema
#         dummy = pa.table({
#             "instance_name": ["dummy"],
#             "k_factor": ["SMALL"],
#             "has_closed_cycle_cycle": [False],
#             "path_vertices": [[0, 1]],
#             "path_length": [0]
#         }, schema=schema)

#         # Write dummy row to create the Delta table
#         write_deltalake(
#             table_or_uri=delta_path,
#             data=dummy,
#             mode="append"
#         )

#         # Overwrite with an empty table having the same schema to remove the dummy row
#         empty_table = pa.table({
#             "instance_name": pa.array([], type=pa.string()),
#             "k_factor": pa.array([], type=pa.string()),
#             "has_closed_cycle_cycle": pa.array([], type=pa.bool_()),
#             "path_vertices": pa.array([], type=pa.list_(pa.int32())),
#             "path_length": pa.array([], type=pa.int32())
#         })

#         write_deltalake(
#             table_or_uri=delta_path,
#             data=empty_table,
#             mode="overwrite",
#             partition_by=["instance_name", "k_factor", "has_closed_cycle_cycle"]
#         )

#         logger.info("Initialized empty Delta table.")

def save_solutions_to_delta(
    solutions: List[Solution],
    delta_path: str
) -> None:
    """
    Given a list of Solution objects, create (if needed) or append 
    a Delta table with the schema:

      instance_name:    string
      k_factor:         string
      has_closed_cycle: bool
      path_vertices:    list<item: int32>
      path_length:      int32

    Each Solution is flattened into exactly those five columns.
    """

    # 1) If Delta folder doesn't exist yet, create an empty Delta with the correct schema
    if not os.path.exists(delta_path):
        # Define Arrow schema (32-bit ints, list<item int32>)
        schema = pa.schema([
            ("instance_name"   , pa.string()),
            ("k_factor"        , pa.string()),
            ("has_closed_cycle", pa.bool_()),
            ("path_vertices"   , pa.list_(pa.int32())),
            ("path_length"     , pa.int32()),
        ])

        # Build an empty Arrow Table of that schema
        empty_tbl = pa.table({
            "instance_name":    pa.array([], type=pa.string()),
            "k_factor":         pa.array([], type=pa.string()),
            "has_closed_cycle": pa.array([], type=pa.bool_()),
            "path_vertices":    pa.array([], type=pa.list_(pa.int32())),
            "path_length":      pa.array([], type=pa.int32()),
        }, schema=schema)

        # Overwrite (create) the Delta folder with an empty table
        write_deltalake(
            table_or_uri=delta_path,
            data=empty_tbl,
            mode="overwrite",
            partition_by=["instance_name", "k_factor", "has_closed_cycle"]
        )
        logger.info(f"Created empty Delta table at {delta_path}.")

    # 2) Flatten each Solution into a dict with exactly the five fields
    instance_names:    List[str] = []
    k_factors:         List[str] = []
    closed_flags:      List[bool] = []
    all_paths:         List[List[int]] = []
    all_lengths:       List[int] = []

    for sol in solutions:
        instance_names.append(sol.instance.name)
        k_factors.append(sol.k_factor.name)
        closed_flags.append(sol.has_closed_cycle)
        all_paths.append(sol.path_vertices)         # Python list[int]
        all_lengths.append(sol.path_length)         # Python int

    # 3) Build Arrow arrays with the correct types
    arr_instance_name = pa.array(instance_names, type=pa.string())
    arr_k_factor      = pa.array(k_factors,      type=pa.string())
    arr_closed        = pa.array(closed_flags,   type=pa.bool_())
    arr_paths         = pa.array(all_paths,      type=pa.list_(pa.int32()))
    arr_lengths       = pa.array(all_lengths,    type=pa.int32())

    arrow_tbl = pa.Table.from_arrays(
        [
            arr_instance_name,
            arr_k_factor,
            arr_closed,
            arr_paths,
            arr_lengths
        ],
        names=[
            "instance_name",
            "k_factor",
            "has_closed_cycle",
            "path_vertices",
            "path_length"
        ]
    )

    # 4) Append into Delta (mode="append")
    write_deltalake(
        table_or_uri=delta_path,
        data=arrow_tbl,
        mode="append"
    )

    logger.info(f"Appended {len(solutions)} distinct solutions to Delta table at {delta_path}.")

def _precompute_one_partition(
    instance_name: str,
    k_factor: KFactor,
    has_closed_cycle: bool,
    delta_path: str
) -> None:
    """
    Computes and saves the top `population_size` NN solutions for the given partition
    using the Experiment abstraction.
    """
    logger.info(f"Starting precompute: instance={instance_name}, k_factor={k_factor}, closed={has_closed_cycle}")
    
    # experiment = Experiment(
    #     experiment_name="nn_precompute",
    #     instance_name=instance_name,
    #     k_factor=k_factor,
    #     has_closed_cycle_cycle=has_closed_cycle,
    #     repetitions=1,
    #     model_name=ModelName.NEAREST_NEIGHBORS_V2.value,
    #     model_parameters={},
    #     delta_path=delta_path
    # )
    try:
        instance = Instance(name=instance_name)
        instance.get_instance()
        initial_population_model = NearestNeighborsV2()
        initial_population = initial_population_model.generate_multiple_solutions(
            instance=instance, 
            k_factor=k_factor,
            has_closed_cycle=has_closed_cycle,
            n_solutions=1000
        )

        save_solutions_to_delta(initial_population, delta_path)

    except Exception as e:
        logger.warning(f"Error processing {instance_name} {k_factor} closed={has_closed_cycle}: {e}")

def precompute_all_nn(
    delta_path: str = "../results/nearest_neighbors_precomputed_solutions" 
):
    tasks = []
    for instance_name in SELECTED_INSTANCES[1:]:
        for k_factor in list(KFactor):
            for has_closed_cycle in [False, True]:
                if (
                    instance_name == "ali535.tsp" 
                    and k_factor == KFactor.SMALL
                ) or (
                    instance_name == "ali535.tsp"
                    and k_factor == KFactor.MEDIUM
                ) or instance_name == "att48.tsp":
                    continue
                tasks.append((instance_name, k_factor, has_closed_cycle, delta_path))

    Parallel(n_jobs=-1, verbose=5)(
        delayed(_precompute_one_partition)(inst, kf, closed, delta_path)
        for inst, kf, closed, delta_path in tasks
    )

    logger.info("All NN partitions have been precomputed.")


precompute_all_nn()