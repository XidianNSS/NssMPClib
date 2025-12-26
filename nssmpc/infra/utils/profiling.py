import time
from contextlib import contextmanager
from functools import wraps

import torch

from nssmpc.infra.utils.common import GREEN, cprint, YELLOW


@contextmanager
def RuntimeTimer(tag: str = None, enable_comm_stats=False):
    """
    Context manager for measuring and printing code block execution time.

    Args:
        tag: Optional identifier for distinguishing different timed sections.
        enable_comm_stats: Optional boolean indicating whether to enable communication stats.

    Examples:
        >>> with RuntimeTimer(tag="function A", enable_comm_stats=True):
        ...     function_a()
        >>> with RuntimeTimer("data_preprocessing"):
        ...     preprocess_data()
        >>> with RuntimeTimer():
        ...     long_running_operation()
    """
    communicator = None
    start_stat = None
    if enable_comm_stats:
        from nssmpc.infra.mpc.party import PartyCtx
        communicator = PartyCtx.get().communicator
        start_stat = communicator.comm_stats.copy()

    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start

    if elapsed < 0.001:
        time_str = f"{elapsed * 1_000_000:.3f}μs"
    elif elapsed < 1:
        time_str = f"{elapsed * 1000:.3f}ms"
    else:
        time_str = f"{elapsed:.3f}s"

    tag_info = f"[{tag}] " if tag else ""
    cprint(f"{tag_info}Execution time: {time_str}", color=GREEN)

    if enable_comm_stats:
        end_stat = communicator.comm_stats
        print(f"Communication costs:\n\tsend rounds: {end_stat['send_count'] - start_stat['send_count']}\t\t"
              f"send bytes: {bytes_convert(end_stat['send_bytes'] - start_stat['send_bytes'])}.")
        print(f"\trecv rounds: {end_stat['recv_count'] - start_stat['recv_count']}\t\t"
              f"recv bytes: {bytes_convert(end_stat['recv_bytes'] - start_stat['recv_bytes'])}.")


def count_bytes(a):
    """
    Calculates the memory footprint of various data structures in bytes.

    Supports torch.Tensor, RingTensor, SecretSharing objects, collections, and primitives.

    Args:
        a (Any): The data structure to measure.

    Returns:
        int: The estimated memory usage in bytes.

    Examples:
        >>> size = count_bytes(my_tensor)
    """
    from nssmpc.infra.tensor import RingTensor
    from nssmpc.primitives.secret_sharing import AdditiveSecretSharing, ReplicatedSecretSharing, BooleanSecretSharing
    if isinstance(a, torch.Tensor):
        return a.element_size() * a.nelement()
    elif isinstance(a, RingTensor):
        return count_bytes(a.tensor)
    elif isinstance(a, (AdditiveSecretSharing, BooleanSecretSharing)):
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
            cprint("Not implemented count_bytes for ", type(a), color=YELLOW, end='')
            if hasattr(a, '__len__'):
                cprint(", Length of the object is ", len(a), color=YELLOW)
            else:
                cprint("", color=YELLOW)
            if hasattr(a, '__dict__'):
                cprint("The bytes of the object might be ", count_bytes(a.__dict__), color=YELLOW)
            return 0


def bytes_convert(byte):
    """
    Converts a byte count into a human-readable string with units (B, KB, MB, GB, TB).

    Args:
        byte (int): The number of bytes.

    Returns:
        str: The formatted string representing the size.

    Examples:
        >>> print(bytes_convert(1024))
    """
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


def run_time(func):
    """
    Decorator that calculates and prints the running time of the decorated function.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: The wrapper function.

    Examples:
        >>> @run_time
        ... def my_func(): pass
    """

    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        cost_time = end - start
        print(f"func {func.__name__} run time {cost_time}")
        return res

    return wrapper


def log(func):
    """
    Decorator that prints a log message before calling the function.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: The wrapper function.

    Examples:
        >>> @log
        ... def my_func(): pass
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"call {func.__name__}()")
        return func(*args, **kwargs)

    return wrapper


def get_time(func, *args):
    """
    Executes a function and prints its execution time.

    Args:
        func (function): The function to call.
        *args: Arguments to pass to the function.

    Returns:
        Any: The result of the function call.

    Examples:
        >>> result = get_time(my_func, arg1, arg2)
    """
    start = time.time()
    res = func(*args)
    end = time.time()
    cprint(f"Time consuming of {func.__name__}: {end - start}", color=GREEN)
    return res


def get_avg_time(func, *args, times=10):
    """
    Calculates and prints the average running time of a function over multiple executions.

    Args:
        func (function): The function to call.
        *args: Arguments to pass to the function.
        times (int, optional): The number of times to run the function. Defaults to 10.

    Examples:
        >>> get_avg_time(my_func, arg1, times=20)
    """
    start = time.time()
    for _ in range(times):
        func(*args)
    end = time.time()
    cprint(f"Average time consuming of {func.__name__}: {(end - start) / times}", color=GREEN)


def comm_count(func, *args, communicator=None):
    """
    Monitors and prints the communication overhead of a function.

    Args:
        communicator (Communicator): The communicator object tracking stats.
        func (function): The function to monitor.
        *args: Arguments to pass to the function.

    Returns:
        Any: The result of the function execution.

    Examples:
        >>> res = comm_count(my_func, arg1)
    """
    if communicator is None:
        from nssmpc.infra.mpc import PartyCtx
        communicator = PartyCtx.get().communicator
    now_comm_rounds = communicator.comm_rounds['send']
    now_comm_bytes = communicator.comm_bytes['send']
    res = func(*args)
    cprint(f"Communication info of {func}:", color=YELLOW)
    cprint(f"Comm_rounds: {communicator.comm_rounds['send'] - now_comm_rounds}", color=YELLOW)
    cprint(f"Comm_costs: {bytes_convert(communicator.comm_bytes['send'] - now_comm_bytes)}", color=YELLOW)
    return res


def statistic(func, *args, times=10, warmup=5):
    """
    Monitors communication overhead and average execution time of a function.

    Args:
        func (function): The function to monitor.
        *args: Arguments to pass to the function.
        times (int, optional): Number of timed runs. Defaults to 10.
        warmup (int, optional): Number of warmup runs before timing. Defaults to 5.

    Returns:
        Any: The result of the first function execution.

    Examples:
        >>> res = statistic(my_func, arg1)
    """
    from nssmpc.infra.mpc.party import PartyCtx
    communicator = PartyCtx.get().communicator
    now_comm_rounds = communicator.comm_rounds['send']
    now_comm_bytes = communicator.comm_bytes['send']
    res = func(*args)
    cprint(f"Communication info of {func}:", color=YELLOW)
    cprint(f"Comm_rounds: {communicator.comm_rounds['send'] - now_comm_rounds}", color=YELLOW)
    cprint(f"Comm_costs: {bytes_convert(communicator.comm_bytes['send'] - now_comm_bytes)}", color=YELLOW)

    while warmup > 0:
        func(*args)
        warmup -= 1

    start = time.time()
    for _ in range(times):
        func(*args)
    end = time.time()
    cprint(f"Average time consuming of {func.__name__}: {(end - start) / times}", color=GREEN)

    return res
