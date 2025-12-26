#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

def list_rotate(list_before, n):
    """
    Rotates a list to the left by n positions.

    Args:
        list_before (list): The list to rotate.
        n (int): The number of positions to shift.

    Returns:
        list: The rotated list.

    Examples:
        >>> new_list = list_rotate([1, 2, 3, 4], 1)
    """
    return [list_before[(i - n) % len(list_before)] for i in range(len(list_before))]


RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
END = '\033[0m'


def cprint(*args, sep=' ', end='\n', file=None, color=GREEN):
    """
    Prints debugging information with a specified color.

    Args:
        *args: Positional arguments to be printed.
        sep (str, optional): String inserted between values, default is a space.
        end (str, optional): String appended after the last value, default is a newline.
        file (file-like object, optional): A file-like object (stream); defaults to the current sys.stdout.
        color (str, optional): ANSI color code for the output text. Defaults to GREEN.

    Examples:
        >>> cprint("Debug info", color=RED)
    """
    print(f"{color}{str(*args)}{END}", sep=sep, end=end, file=file)


def align_shape(x_shape, y_shape):
    x_shape = list(x_shape)
    y_shape = list(y_shape)
    if len(x_shape) > len(y_shape):
        y_shape = [1] * (len(x_shape) - len(y_shape)) + y_shape
    elif len(x_shape) < len(y_shape):
        x_shape = [1] * (len(y_shape) - len(x_shape)) + x_shape

    i = 0
    while x_shape[i] == 1 and y_shape[i] == 1 and i < len(x_shape):
        i += 1

    return x_shape[i:], y_shape[i:]
