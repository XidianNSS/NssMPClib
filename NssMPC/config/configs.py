#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import json
import os
from pathlib import Path

import torch

CONFIG_DIR = Path(__file__).parent
CONFIG_FILE = CONFIG_DIR / 'config.json'
DEFAULT_CONFIG = CONFIG_DIR / 'default_config.json'

if not CONFIG_FILE.exists():
    if DEFAULT_CONFIG.exists():
        CONFIG_FILE.write_bytes(DEFAULT_CONFIG.read_bytes())
        print("未找到 config.json，已从 default_config.json 重新创建。")
    else:
        raise FileNotFoundError("缺少配置文件")

with open(CONFIG_FILE, 'r') as f:
    config = json.load(f)

BIT_LEN = config['BIT_LEN']
LAMBDA = config['LAMBDA']
PRG_TYPE = config['PRG_TYPE']
DEBUG_LEVEL = config['DEBUG_LEVEL']
DTYPE = config['DTYPE']
SCALE_BIT = config['SCALE_BIT']
SOCKET_TYPE = config['SOCKET_TYPE']  # 0: Normal   1: Multi-threaded
EXP_ITER = config['EXP_ITER']
GELU_TABLE_BIT = config['GELU_TABLE_BIT']
TANH_TABLE_BIT = config['TANH_TABLE_BIT']
VCMP_SPLIT_LEN = config['VCMP_SPLIT_LEN']
SOCKET_MAX_SIZE = config['SOCKET_MAX_SIZE']
SOCKET_NUM = config['SOCKET_NUM']
DEVICE = os.environ.get('DEVICE') if os.environ.get('DEVICE') else config['DEVICE']

base_path = Path(__file__).parent.parent.parent
param_path = base_path / f'data/{str(BIT_LEN)}/aux_parameters/'
NN_path = base_path / 'data/NN/'

RING_MAX = 2 ** BIT_LEN
HALF_RING = 2 ** (BIT_LEN - 1)
data_type = torch.int64 if BIT_LEN == 64 else torch.int32  # Only support BIT_LEN = 64 or 32
# Fixed Point Setting
float_scale = 2 ** SCALE_BIT
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

assert DEBUG_LEVEL in (0, 1, 2)
assert BIT_LEN in (64, 32)

SOCKET_P0 = config['SOCKET_P0']
SOCKET_P1 = config['SOCKET_P1']
SOCKET_P2 = config['SOCKET_P2']

SOCKET_P0['FROM_NEXT'] = SOCKET_P1['TO_PREVIOUS']
SOCKET_P0['FROM_PREVIOUS'] = SOCKET_P2['TO_NEXT']
SOCKET_P0['ADDRESS_NEXT'] = SOCKET_P1['ADDRESS']
SOCKET_P0['PORT_NEXT'] = SOCKET_P1['PORT']
SOCKET_P0['ADDRESS_PREVIOUS'] = SOCKET_P2['ADDRESS']
SOCKET_P0['PORT_PREVIOUS'] = SOCKET_P2['PORT']

SOCKET_P1['FROM_NEXT'] = SOCKET_P2['TO_PREVIOUS']
SOCKET_P1['FROM_PREVIOUS'] = SOCKET_P0['TO_NEXT']
SOCKET_P1['ADDRESS_NEXT'] = SOCKET_P2['ADDRESS']
SOCKET_P1['PORT_NEXT'] = SOCKET_P2['PORT']
SOCKET_P1['ADDRESS_PREVIOUS'] = SOCKET_P0['ADDRESS']
SOCKET_P1['PORT_PREVIOUS'] = SOCKET_P0['PORT']

SOCKET_P2['FROM_NEXT'] = SOCKET_P0['TO_PREVIOUS']
SOCKET_P2['FROM_PREVIOUS'] = SOCKET_P1['TO_NEXT']
SOCKET_P2['ADDRESS_NEXT'] = SOCKET_P0['ADDRESS']
SOCKET_P2['PORT_NEXT'] = SOCKET_P0['PORT']
SOCKET_P2['ADDRESS_PREVIOUS'] = SOCKET_P1['ADDRESS']
SOCKET_P2['PORT_PREVIOUS'] = SOCKET_P1['PORT']
