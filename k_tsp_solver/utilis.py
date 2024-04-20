import time
from typing import Callable, Any


def timeit(func: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {round(end_time - start_time, 2)} sec to execute.") 

        return result
    
    return wrapper