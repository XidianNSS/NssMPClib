#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

def list_rotate(list_before, n):
    """
    Rotate the list list_before to the left by n positions.

    For each index item in the list, the *(i-n) % len(list_before)* operation is performed to calculate the rotated position

    :param list_before: A list of actions to perform
    :type list_before: list
    :param n: Moved bits
    :type n: int
    :return: The list of completed rotations to the left
    :rtype: list
    """
    return [list_before[(i - n) % len(list_before)] for i in range(len(list_before))]
