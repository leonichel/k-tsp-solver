import time
from enum import Enum
from typing import Callable, Any

from deltalake import DeltaTable
import pandas as pd

from k_tsp_solver import logger


DELTA_PATH = "experiments/"


def timeit(func: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Function {func.__name__} took {round(end_time - start_time, 2)} sec to execute.") 

        return result
    
    return wrapper

def dataclass_to_dict(dataclass_instance: Any) -> dict:
    return {
        field.name: getattr(dataclass_instance, field.name)
            if not isinstance(getattr(dataclass_instance, field.name), Enum)
            else getattr(dataclass_instance, field.name).value
        for field in dataclass_instance.__dataclass_fields__.values()
        if field.repr
    }

def read_experiments() -> pd.DataFrame:
    return DeltaTable(DELTA_PATH).to_pandas()