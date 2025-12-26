from nssmpc.protocols.honest_majority_3pc import *
from nssmpc.protocols.semi_honest_2pc import *
from nssmpc.protocols.semi_honest_3pc import *

SEMI_HONEST = {
    'additive': {
        '_add_public_value': sh2pc_add_public_value,
        '_trunc': sh2pc_truncate,
        '_mul': sh2pc_mul_beaver,
        '_matmul': sh2pc_matmul_beaver,
        '_ge': sh2pc_ge_sigma,
        '_eq': sh2pc_eq,
        '_div': sh2pc_div,
        'exp': sh2pc_exp,
        'rsqrt': sh2pc_reciprocal_sqrt,
        'tanh': sh2pc_tanh,
        'recon': sh2pc_recon
    },
    'boolean': {
        '_and': sh2pc_beaver_and
    },
    'replicated': {
        '_add_public_value': sh3pc_add_public_value,
        '_mul': sh3pc_mul,
        '_matmul': sh3pc_matmul,
        '_ge': sh3pc_ge,
        '_trunc': hm3pc_truncate_aby3,
        'recon': sh3pc_recon
    }
}

HONEST_MAJORITY = {
    'additive': {
        '_add_public_value': sh2pc_add_public_value,
        '_trunc': sh2pc_truncate,
        '_mul': sh2pc_mul_beaver,
        '_matmul': sh2pc_matmul_beaver,
        '_ge': sh2pc_ge_sigma,
        '_eq': sh2pc_eq,
        '_div': sh2pc_div,
        'exp': sh2pc_exp,
        'rsqrt': sh2pc_reciprocal_sqrt,
        'tanh': sh2pc_tanh,
        'recon': sh2pc_recon
    },
    'boolean': {
        '_and': sh2pc_beaver_and
    },
    'replicated': {
        '_add_public_value': sh3pc_add_public_value,
        '_mul': hm3pc_mul,
        '_matmul': hm3pc_matmul,
        '_ge': hm3pc_ge,
        '_trunc': hm3pc_truncate_aby3,
        'recon': hm3pc_recon
    }
}
