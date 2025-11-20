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
