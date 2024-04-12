from common.tensor.ring_tensor import RingTensor
from config.base_configs import *


def where(condition, x, y):
    if isinstance(x, RingTensor) and isinstance(y, RingTensor):
        return RingTensor(torch.where(condition.tensor.bool(), x.tensor, y.tensor), x.dtype, x.device)
    elif isinstance(x, RingTensor) and isinstance(y, int):
        return RingTensor(torch.where(condition.tensor.bool(), x.tensor, y), x.dtype, x.device)
    elif isinstance(x, int) and isinstance(y, RingTensor):
        return RingTensor(torch.where(condition.tensor.bool(), x, y.tensor), y.dtype, y.device)
    elif isinstance(x, int) and isinstance(y, int):
        return RingTensor(torch.where(condition.tensor.bool(), x, y).to(data_type), condition.dtype, condition.device)


def random(shape, dtype='int', device=DEVICE, down_bound=-HALF_RING, upper_bound=HALF_RING - 1):
    v = torch.randint(down_bound, upper_bound, shape, dtype=data_type, device=device)
    return RingTensor(v, dtype, device)


def empty(size, dtype='int', device=DEVICE):
    return RingTensor(torch.empty(size, dtype=data_type), dtype, device)


def empty_like(tensor):
    if isinstance(tensor, RingTensor):
        return RingTensor(torch.empty_like(tensor.tensor), tensor.dtype, tensor.device)
    else:
        raise TypeError("unsupported operand type(s) for empty_like 'RingTensor' and ", type(tensor))


def zeros(size, dtype='int', device=DEVICE):
    return RingTensor(torch.zeros(size, dtype=data_type, device=device), dtype)


def zeros_like(tensor, dtype='int', device=DEVICE):
    if isinstance(tensor, RingTensor):
        return RingTensor(torch.zeros_like(tensor.tensor), tensor.dtype, tensor.device)
    elif isinstance(tensor, torch.Tensor):
        return RingTensor(torch.zeros_like(tensor), dtype, device)
    else:
        raise TypeError("unsupported operand type(s) for zeros_like 'RingTensor' and ", type(tensor))


def ones(size, dtype='int', device=DEVICE):
    scale = DTYPE_SCALE_MAPPING[dtype]
    return RingTensor(torch.ones(size, dtype=data_type, device=device) * scale, dtype)


def ones_like(tensor, dtype='int', device=DEVICE):
    if isinstance(tensor, RingTensor):
        return RingTensor(torch.ones_like(tensor.tensor) * tensor.scale, tensor.dtype, tensor.device)
    elif isinstance(tensor, torch.Tensor):
        scale = DTYPE_SCALE_MAPPING[dtype]
        return RingTensor(torch.ones_like(tensor) * scale, dtype, device)
    else:
        raise TypeError("unsupported operand type(s) for ones_like 'RingTensor' and ", type(tensor))


def full(size, fill_value, device=DEVICE):
    return RingTensor.convert_to_ring(torch.full(size, fill_value, device=device))


def full_like(tensor, fill_value, device=DEVICE):
    if isinstance(tensor, RingTensor):
        return full(tensor.shape, fill_value, tensor.device)
    elif isinstance(tensor, torch.Tensor):
        return RingTensor.convert_to_ring(torch.full_like(tensor, fill_value, device=device))
    else:
        raise TypeError("unsupported operand type(s) for full_like 'RingTensor' and ", type(tensor))


def diagonal(input, offset=0, dim1=0, dim2=1):
    return RingTensor(torch.diagonal(input.tensor, offset, dim1, dim2), input.dtype, input.device)


def arange(start, end, step=1, dtype='int', device=DEVICE):
    return RingTensor(torch.arange(start, end, step, dtype=data_type, device=device), dtype)
