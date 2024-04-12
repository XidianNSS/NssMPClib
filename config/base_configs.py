import torch

BIT_LEN = 64
RING_MAX = 2 ** BIT_LEN
HALF_RING = 2 ** (BIT_LEN - 1)
LAMBDA = 128

GE_TYPE = 'SIGMA'
PRG_TYPE = 'AES'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

DEBUG_LEVEL = 0
assert DEBUG_LEVEL in (0, 1, 2)
assert BIT_LEN in (64, 32)
assert GE_TYPE in ('MSB', 'DICF', 'PPQ', 'SIGMA')
assert DEVICE in ('cuda', 'cpu', 'cuda:0', 'cuda:1')
data_type = torch.int64 if BIT_LEN == 64 else torch.int32  # Only support BIT_LEN = 64 or 32

# Fixed Point Setting
DTYPE = 'float'
# The precision of floating point numbers on 64-bit rings is 65536, and the precision on 32-bit rings is 128
float_scale = 65536 if BIT_LEN == 64 else 128
int_scale = 1
SCALE = float_scale if DTYPE == 'float' else int_scale

DTYPE_MAPPING = {
    torch.int32: int_scale,
    torch.int64: int_scale,
    torch.float32: float_scale,
    torch.float64: float_scale
}

DTYPE_SCALE_MAPPING = {
    'int': int_scale,
    'float': float_scale,
}

base_path = f'./data/{BIT_LEN}'
model_file_path = base_path + '/neural_network/'
