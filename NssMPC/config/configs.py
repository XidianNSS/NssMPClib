#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import json
import os

import torch
from NssMPC.config import __data_path

# root path
base_path = __data_path

# config path
config_path = base_path + 'config.json'
with open(config_path, 'r') as config_file:
    config = json.load(config_file)

BIT_LEN = config['BIT_LEN']
LAMBDA = config['LAMBDA']
GE_TYPE = config['GE_TYPE']
PRG_TYPE = config['PRG_TYPE']
DEVICE = os.environ.get('DEVICE') if os.environ.get('DEVICE') else config['DEVICE']
DEBUG_LEVEL = config['DEBUG_LEVEL']
DTYPE = config['DTYPE']
SOCKET_TYPE = config['SOCKET_TYPE']  # 0: Normal   1: Multi-threaded

param_path = base_path + 'data/' + str(BIT_LEN) + '/aux_parameters/'
NN_path = base_path + 'data/NN/'
RING_MAX = 2 ** BIT_LEN
HALF_RING = 2 ** (BIT_LEN - 1)
assert DEBUG_LEVEL in (0, 1, 2)
assert BIT_LEN in (64, 32)
assert GE_TYPE in ('MSB', 'DICF', 'PPQ', 'SIGMA')
# assert DEVICE in ('cuda', 'cpu', 'cuda:0', 'cuda:1')
data_type = torch.int64 if BIT_LEN == 64 else torch.int32  # Only support BIT_LEN = 64 or 32

# Fixed Point Setting
SCALE_BIT = config['SCALE_BIT']
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

EXP_ITER = config['EXP_ITER']

GELU_TABLE_BIT = config['GELU_TABLE_BIT']
TANH_TABLE_BIT = config['TANH_TABLE_BIT']

VCMP_SPLIT_LEN = config['VCMP_SPLIT_LEN']

SOCKET_MAX_SIZE = config['SOCKET_MAX_SIZE']
SOCKET_NUM = config['SOCKET_NUM']


class Config:
    """
    The class serves as a tool for configuration management.
    """

    def __init__(self, cfg=None):
        """
        It accepts an optional dictionary ``cfg`` during initialization, and then uses ``self.__dict__.update(cfg)`` to update
        the key-value pairs in the dictionary to the instance attributes of the class. In this way, each key in the
        dictionary becomes a property of the class, making it convenient to access and manage the configuration later on.

        .. note::
            For example, if you pass in a dictionary ``{'BIT_LEN': 128, 'LAMBDA': 0.01}``, you can access these values by using ``config_instance.BIT_LEN`` and ``config_instance.LAMBDA``.

        :param cfg: an optional dictionary to initialize class
        :type cfg: dict
        """
        self.__dict__.update(cfg)


# SOCKET_3PC = Config(config['SOCKET_3PC'])
# SOCKET_2PC = Config(config['SOCKET_2PC'])

SOCKET_P0 = Config(config['SOCKET_P0'])
SOCKET_P1 = Config(config['SOCKET_P1'])
SOCKET_P2 = Config(config['SOCKET_P2'])

SOCKET_P0.FROM_NEXT = SOCKET_P1.TO_PREVIOUS
SOCKET_P0.FROM_PREVIOUS = SOCKET_P2.TO_NEXT
SOCKET_P0.ADDRESS_NEXT = SOCKET_P1.ADDRESS
SOCKET_P0.PORT_NEXT = SOCKET_P1.PORT
SOCKET_P0.ADDRESS_PREVIOUS = SOCKET_P2.ADDRESS
SOCKET_P0.PORT_PREVIOUS = SOCKET_P2.PORT

SOCKET_P1.FROM_NEXT = SOCKET_P2.TO_PREVIOUS
SOCKET_P1.FROM_PREVIOUS = SOCKET_P0.TO_NEXT
SOCKET_P1.ADDRESS_NEXT = SOCKET_P2.ADDRESS
SOCKET_P1.PORT_NEXT = SOCKET_P2.PORT
SOCKET_P1.ADDRESS_PREVIOUS = SOCKET_P0.ADDRESS
SOCKET_P1.PORT_PREVIOUS = SOCKET_P0.PORT

SOCKET_P2.FROM_NEXT = SOCKET_P0.TO_PREVIOUS
SOCKET_P2.FROM_PREVIOUS = SOCKET_P1.TO_NEXT
SOCKET_P2.ADDRESS_NEXT = SOCKET_P0.ADDRESS
SOCKET_P2.PORT_NEXT = SOCKET_P0.PORT
SOCKET_P2.ADDRESS_PREVIOUS = SOCKET_P1.ADDRESS
SOCKET_P2.PORT_PREVIOUS = SOCKET_P1.PORT
