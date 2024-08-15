def list_rotate(list_before, n):
    return [list_before[(i - n) % len(list_before)] for i in range(len(list_before))]
