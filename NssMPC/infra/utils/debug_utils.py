#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

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
    """
    Prints debugging information with a specified color.

    Args:
        *args: Positional arguments to be printed.
        sep (str, optional): String inserted between values, default is a space.
        end (str, optional): String appended after the last value, default is a newline.
        file (file-like object, optional): A file-like object (stream); defaults to the current sys.stdout.
        color (str, optional): ANSI color code for the output text. Defaults to GREEN.

    Examples:
        >>> debug_print("Debug info", color=RED)
    """
    print(f"{color}{str(*args)}{END}", sep=sep, end=end, file=file)


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
    debug_print(f"Time consuming of {func.__name__}: {end - start}", color=GREEN)
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
    debug_print(f"Average time consuming of {func.__name__}: {(end - start) / times}", color=GREEN)


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
    from NssMPC.infra.tensor import RingTensor
    from NssMPC.primitives.secret_sharing import AdditiveSecretSharing, ReplicatedSecretSharing, BooleanSecretSharing
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
            debug_print("Not implemented count_bytes for ", type(a), color=YELLOW, end='')
            if hasattr(a, '__len__'):
                debug_print(", Length of the object is ", len(a), color=YELLOW)
            else:
                debug_print("", color=YELLOW)
            if hasattr(a, '__dict__'):
                debug_print("The bytes of the object might be ", count_bytes(a.__dict__), color=YELLOW)
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


def comm_count(communicator, func, *args):
    """
    Monitors and prints the communication overhead of a function.

    Args:
        communicator (Communicator): The communicator object tracking stats.
        func (function): The function to monitor.
        *args: Arguments to pass to the function.

    Returns:
        Any: The result of the function execution.

    Examples:
        >>> res = comm_count(comm, my_func, arg1)
    """
    now_comm_rounds = communicator.comm_rounds['send']
    now_comm_bytes = communicator.comm_bytes['send']
    res = func(*args)
    debug_print(f"Communication info of {func}:", color=YELLOW)
    debug_print(f"Comm_rounds: {communicator.comm_rounds['send'] - now_comm_rounds}", color=YELLOW)
    debug_print(f"Comm_costs: {bytes_convert(communicator.comm_bytes['send'] - now_comm_bytes)}", color=YELLOW)
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
    from NssMPC.infra.mpc.party import PartyCtx
    communicator = PartyCtx.get().communicator
    now_comm_rounds = communicator.comm_rounds['send']
    now_comm_bytes = communicator.comm_bytes['send']
    res = func(*args)
    debug_print(f"Communication info of {func}:", color=YELLOW)
    debug_print(f"Comm_rounds: {communicator.comm_rounds['send'] - now_comm_rounds}", color=YELLOW)
    debug_print(f"Comm_costs: {bytes_convert(communicator.comm_bytes['send'] - now_comm_bytes)}", color=YELLOW)

    while warmup > 0:
        func(*args)
        warmup -= 1

    start = time.time()
    for _ in range(times):
        func(*args)
    end = time.time()
    debug_print(f"Average time consuming of {func.__name__}: {(end - start) / times}", color=GREEN)

    return res
