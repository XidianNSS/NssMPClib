from NssMPC.protocols.honest_majority_3pc import (
    multiplication as hm3pc_mul,
    compare as hm3pc_cmp,
)
from NssMPC.protocols.semi_honest_2pc import (
    truncate as sh2pc_trunc,
    secure_exp,
    secure_tanh,
    comparison,
    division,
    multiplication,
    boolean_and,
    reciprocal_sqrt
)
from NssMPC.protocols.semi_honest_2pc.base import add_public_value as add_public_value_2pc
from NssMPC.protocols.semi_honest_3pc import (
    comparison as sh3pc_cmp,
    multiplication as sh3pc_mul,
    truncate as sh3pc_trunc,
)
from NssMPC.protocols.semi_honest_3pc.base import add_public_value as add_public_value_3pc

SEMI_HONEST = {
    'additive': {
        '_add_public_value': add_public_value_2pc,
        '_trunc': sh2pc_trunc,
        '_mul': multiplication.beaver_mul,
        '_matmul': multiplication.secure_matmul,
        '_ge': comparison.sigma_ge,
        '_eq': comparison.secure_eq,
        '_div': division.secure_div,
        'exp': secure_exp,
        'rsqrt': reciprocal_sqrt.secure_reciprocal_sqrt,
        'tanh': secure_tanh
    },
    'boolean': {
        '_and': boolean_and.beaver_and
    },
    'replicated': {
        '_add_public_value': add_public_value_3pc,
        '_mul': sh3pc_mul.mul,
        '_matmul': sh3pc_mul.matmul,
        '_ge': sh3pc_cmp.secure_ge,
        '_trunc': sh3pc_trunc.truncate
    }
}

HONEST_MAJORITY = {
    'additive': {
        '_add_public_value': add_public_value_3pc,
        '_trunc': sh2pc_trunc,
        '_mul': multiplication.beaver_mul,
        '_matmul': multiplication.secure_matmul,
        '_ge': comparison.sigma_ge,
        '_eq': comparison.secure_eq,
        '_div': division.secure_div,
        'exp': secure_exp,
        'rsqrt': reciprocal_sqrt.secure_reciprocal_sqrt,
        'tanh': secure_tanh
    },
    'boolean': {
        '_and': boolean_and.beaver_and
    },
    'replicated': {
        '_add_public_value': add_public_value_3pc,
        '_mul': hm3pc_mul.v_mul,
        '_matmul': hm3pc_mul.v_matmul,
        '_ge': hm3pc_cmp.secure_ge,
        '_trunc': sh3pc_trunc.truncate
    }
}
