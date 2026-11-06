"""Trusted preprocessing for local tests sharing a parameter directory.

Only P0 writes parameter files. All parties wait until those files are ready.
This test dealer is not a preprocessing protocol for private deployments.
"""

from nssmpc import HONEST_MAJORITY, Party2PC, Party3PC
from nssmpc.application.neural_network.utils.beaver import gen_mat_beaver
from nssmpc.protocols.honest_majority_3pc.multiplication import RssMatmulTriples
from nssmpc.protocols.semi_honest_2pc.multiplication import MatmulTriples


def prepare_matrix_triples(party, x_shape, y_shape, num_of_triples=1):
    """Generate the matrix triples needed by a local test before their first use."""
    if party.party_id == 0:
        if isinstance(party, Party2PC):
            MatmulTriples.gen_and_save(num_of_triples, x_shape, y_shape)
        elif isinstance(party, Party3PC) and party.thread_model_cfg == HONEST_MAJORITY:
            RssMatmulTriples.gen_and_save(num_of_triples, x_shape, y_shape)
    party.communicator.barrier()


def prepare_model_triples(party, model=None, dummy_input=None, num_of_triples=1):
    """Generate model-shaped triples on P0 and synchronize local test parties.

    P0 supplies the plaintext model and a dummy input with the public test shape.
    Other parties need neither. Call this before loading any matrix parameters.
    """
    if party.party_id == 0:
        gen_mat_beaver(dummy_input, model, num_of_triples, party, save=True)
    party.communicator.barrier()
