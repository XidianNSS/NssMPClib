#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import time
from functools import wraps
import torch

from NssMPC.config.runtime import PartyRuntime

RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
END = '\033[0m'


def debug_print(*args, sep=' ', end='\n', file=None, color=GREEN):
    """
    Print debugging information with specified color.
    The input parameters, except `color`, have the same purpose and type as the ``print`` function.
    """
    print(f"{color}{str(*args)}{END}", sep=sep, end=end, file=file)


def run_time(func):
    """
    Calculates the running time of the decorated function.

    First, record the start time before calling the function. Then, Call the function and record the return value. Finally, Calculate and print run time.

    :param func: Function to be decorated
    :type func: function
    :return: run time
    :rtype: time
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
    Print the log, and keep the documentation and name information of the original function.

    :param func: Function to be decorated
    :type func: function
    :return: log record
    :rtype: log
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"call {func.__name__}()")
        return func(*args, **kwargs)

    return wrapper


def get_time(func, *args):
    """
    Get the run time of the function. Besides, the return is same as the func.

    .. note::
        The difference with :func:`run_time` is that *run_time* is invoked through a decorator.

    :param func: the function that will be called
    :type func: function
    :param args: the arguments to be passed to the function
    :type args: Any
    :return: the result of the input function.
    :rtype: Any
    """
    start = time.time()
    res = func(*args)
    end = time.time()
    debug_print(f"Time consuming of {func.__name__}: {end - start}", color=GREEN)
    return res


def get_avg_time(func, *args, times=10):
    """
    Get the average run time of the function

    :param func: the function that will be called
    :type func: function
    :param args: the arguments to be passed to the function
    :type args: list
    :param times: the number of times to run the function (Default is 10)
    :type times: int
    """
    start = time.time()
    for _ in range(times):
        func(*args)
    end = time.time()
    debug_print(f"Average time consuming of {func.__name__}: {(end - start) / times}", color=GREEN)


def count_bytes(a):
    """
    The function is used to calculate the memory footprint of different data structures.

    **First, determine the type of input parameters:**
        * If the input is a **torch.Tensor**:
            Calculate its byte size.(*element_size()* returns the number of bytes for each element. *nelement()* returns the total number of elements)
        * If the input is a **RingTensor**:
            The :func:`count_bytes` function is recursively called to calculate the number of bytes in its internal tensors.
        * If the input is a **ArithmeticSecretSharing** or a **BooleanSecretSharing**:
            Recursive calls to count_bytes are made to calculate the byte count of its items.
        * If the input is a **ReplicatedSecretSharing**:
            The number of bytes called for the first item, multiplied by 2. (Because it has two copies.)

    **Processing set type:**
        * If the input is a **list** or **tuple**:
            The generator expression recursively calculates the byte count for each element and sums them up.
        * If the input is a **dictionary**:
            Recursively calculate the byte count of all values and sum them up.

    **Primary data type:**
        * If the input is a **bool** type object:
            Boolean values take up 1 byte.
        * If the input is a **int** type object:
            An integer takes up 4 bytes.
        * If the input is a **float** type object:
            Floating-point types take up 8 bytes.
        * If the input is a **string** type object:
            The number of bytes in a string is equal to its length.

    Other types:
        * For **non-directly supported types**:
            Check whether the object has a *__dict__* attribute (indicating that it is an object), and if so, recursively calculate the byte count of its attributes.
        * If the type is **not supported**:
            Print debugging information, including the type and length (if applicable), and the possible number of bytes.

    :param a: The data structure to be computed
    :type a: Any
    :return: The amount of memory used by data structures
    :rtype: int
    """
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
    """
    Convert bytes to a more readable format (B, KB, MB, GB, TB).

    There are five cases:
        * If byte is less than 1024, return the number of bytes and the unit "B".
        * If it is between 1024 and 1MB, return the number of kilobytes and the unit "KB".
        * If it is between 1MB and 1GB, return the number of megabytes and the unit "MB".
        * If it is between 1GB and 1TB, return the number of gigabytes and the unit "GB".
        * If greater than or equal to 1TB, return terabytes and the unit "TB".

    :param byte: number of bytes
    :type byte: int
    :return: The number of bytes after conversion
    :rtype: str
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
    Monitor the communication overhead of specific functions, including the number of communication rounds and the number of bytes sent.

    :param communicator: Communicating party
    :type communicator: Communicator
    :param func: monitored function
    :type func: function
    :param args: Incoming parameter
    :type args: Any
    :return: The execution result of the function
    :rtype: Any
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
    Monitor the communication overhead of specific functions,
    including the number of communication rounds, the number of bytes sent and recv and the average time-consuming.
    :param func:
    :param args:
    :param times:
    :param warmup:
    :return:
    """
    communicator = PartyRuntime.party.communicator
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
