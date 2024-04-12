import time
from functools import wraps

import torch
from tqdm import tqdm

RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
END = '\033[0m'


def debug_print(*args, sep=' ', end='\n', file=None, color=GREEN):
    print(f"{color}{str(*args)}{END}", sep=sep, end=end, file=file)


def run_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        cost_time = end - start
        print(f"func {func.__name__} run time {cost_time}")
        return res

    return wrapper


def log(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"call {func.__name__}()")
        return func(*args, **kwargs)

    return wrapper


def get_time(func, *args):
    """
    Get the run time of the function.
    Besides, the return is same as the func.

    Args:
        func (function): the function that will be called
        *args (list): the arguments to be passed to the function

    Returns:
        the return of the func
    """
    start = time.time()
    res = func(*args)
    end = time.time()
    debug_print(f"Time consuming of {func.__name__}: {end - start}", color=GREEN)
    return res


def get_avg_time(func, *args, times=10):
    """
    Get the average run time of the function

    Args:
        func (function): the function that will be called
        *args (list): the arguments to be passed to the function
        times (int): the number of times to run the function
    """
    start = time.time()
    for _ in range(times):
        func(*args)
    end = time.time()
    debug_print(f"Average time consuming of {func.__name__}: {(end - start) / times}", color=GREEN)


def get_time_by_tqdm(func, *args, times=10):
    """
    Get the average run time of the function.
    And you can see the progress bar by using tqdm.

    Args:
        func (function): the function that will be called
        *args (list): the arguments to be passed to the function
        times (int): the number of times to run the function
    """
    start = time.time()
    for _ in tqdm(range(times)):
        func(*args)
    end = time.time()
    debug_print(f"Average time consuming of {func.__name__}: {(end - start) / times}", color=GREEN)


def count_bytes(a):
    # TODO: Calculation of dictionary and other types
    if isinstance(a, torch.Tensor):
        return a.element_size() * a.nelement()
    elif isinstance(a, dict):
        return 0
    elif isinstance(a, int):
        return 4
    elif isinstance(a, float):
        return 8
    elif isinstance(a, str):
        return 0
    else:
        return 0


def bytes_convert(byte):
    if byte < 1024:
        return f"{byte} B"
    elif byte < 1024 ** 2:
        return f"{byte / 1024} KB"
    elif byte < 1024 ** 3:
        return f"{byte / 1024 ** 2} MB"
    elif byte < 1024 ** 4:
        return f"{byte / 1024 ** 3} GB"
    else:
        return f"{byte / 1024 ** 4} TB"


def comm_count(communicator, func, *args):
    now_comm_rounds = communicator.comm_rounds['send']
    now_comm_bytes = communicator.comm_bytes['send']
    func(*args)
    debug_print(f"Communication info of {func.__name__}:", color=YELLOW)
    debug_print(f"Comm_rounds: {communicator.comm_rounds['send'] - now_comm_rounds}", color=YELLOW)
    debug_print(f"Comm_costs: {bytes_convert(communicator.comm_bytes['send'] - now_comm_bytes)}", color=YELLOW)
