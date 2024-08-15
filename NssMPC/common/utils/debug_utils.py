import time
from functools import wraps
import torch

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
        print(f"func {func} run time {cost_time}")
        return res

    return wrapper


def log(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"call {func}()")
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
    debug_print(f"Time consuming of {func}: {end - start}", color=GREEN)
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
    debug_print(f"Average time consuming of {func}: {(end - start) / times}", color=GREEN)


def count_bytes(a):
    from NssMPC.common.ring import RingTensor
    from NssMPC.crypto.primitives.arithmetic_secret_sharing import ArithmeticSecretSharing, ReplicatedSecretSharing
    from NssMPC.crypto.primitives.boolean_secret_sharing import BooleanSecretSharing

    if isinstance(a, torch.Tensor):
        return a.element_size() * a.nelement()
    elif isinstance(a, RingTensor):
        return count_bytes(a.tensor)
    elif isinstance(a, (ArithmeticSecretSharing, BooleanSecretSharing)):
        return count_bytes(a.item)
    elif isinstance(a, ReplicatedSecretSharing):
        return count_bytes(a.item[0]) * 2
    elif isinstance(a, (list, tuple)):
        return sum(count_bytes(i) for i in a)
    elif isinstance(a, dict):
        return sum(count_bytes(i) for i in a.values())
    elif isinstance(a, bool):
        return 1
    elif isinstance(a, int):
        return 4
    elif isinstance(a, float):
        return 8
    elif isinstance(a, str):
        return len(a)
    else:
        if hasattr(a, '__dict__'):
            return count_bytes(a.__dict__)
        else:
            debug_print("Not implemented count_bytes for ", type(a), color=YELLOW, end='')
            if hasattr(a, '__len__'):
                debug_print(", Length of the object is ", len(a), color=YELLOW)
            else:
                debug_print("", color=YELLOW)
            if hasattr(a, '__dict__'):
                debug_print("The bytes of the object might be ", count_bytes(a.__dict__), color=YELLOW)
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
    res = func(*args)
    debug_print(f"Communication info of {func}:", color=YELLOW)
    debug_print(f"Comm_rounds: {communicator.comm_rounds['send'] - now_comm_rounds}", color=YELLOW)
    debug_print(f"Comm_costs: {bytes_convert(communicator.comm_bytes['send'] - now_comm_bytes)}", color=YELLOW)
    return res
