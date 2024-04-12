from common.aux_parameter.param_provider import ParamProvider
from common.tensor import *
from config.base_configs import DTYPE, base_path, DEBUG_LEVEL
from crypto.primitives.beaver.beaver_triples import BeaverTriples


class MatrixTriples(BeaverTriples):
    @staticmethod
    def gen(num_of_triples, x_shape=None, y_shape=None, num_of_party=2):
        """
        Generate multiplicative Beaver triples

        Args:
            num_of_triples: the number of triples
            num_of_party: the number of parties
            x_shape: the shape of the matrix x
            y_shape: the shape of the matrix y
        """
        return gen_matrix_triples_by_ttp(num_of_triples, x_shape, y_shape, num_of_party)

    @classmethod
    def gen_and_save(cls, num_of_triples, x_shape=None, y_shape=None, num_of_party=2):
        """
        Generate and save multiplicative Beaver triples

        Args:
            num_of_triples: the number of triples
            num_of_party: the number of parties
            x_shape: the shape of the matrix x
            y_shape: the shape of the matrix y
        """
        triples = cls.gen(num_of_triples, x_shape, y_shape, num_of_party)
        for party_id in range(num_of_party):
            file_path = base_path + f"/aux_parameters/BeaverTriples/{num_of_party}party/Matrix"
            file_name = f"MatrixBeaverTriples_{party_id}_{list(x_shape)}_{list(y_shape)}.pth"
            triples[party_id].save_by_name(file_name, file_path)


def gen_matrix_triples_by_ttp(num_of_param, x_shape, y_shape, num_of_party=2):
    """
    Generate the matrix multiplication Beaver triple by trusted third party

    Args:
        num_of_param: the number of matrix Beaver triples to need
        x_shape: the shape of the matrix x
        y_shape: the shape of the matrix y
        num_of_party: the number of party

    Returns:
        the matrix multiplication Beaver triples
    """
    x_shape = [num_of_param] + list(x_shape)
    y_shape = [num_of_param] + list(y_shape)
    a = RingFunc.random(x_shape)
    b = RingFunc.random(y_shape)
    if a.device == 'cpu':
        c = a @ b
    else:
        c = cuda_matmul(a.tensor, b.tensor)
        c = RingTensor.convert_to_ring(c)

    from crypto.primitives.arithmetic_secret_sharing.arithmetic_shared_ring_tensor import ArithmeticSharedRingTensor
    a_list = ArithmeticSharedRingTensor.share(a, num_of_party)
    b_list = ArithmeticSharedRingTensor.share(b, num_of_party)
    c_list = ArithmeticSharedRingTensor.share(c, num_of_party)

    triples = []
    for i in range(num_of_party):
        triples.append(MatrixTriples())
        triples[i].a = a_list[i].to('cpu')
        triples[i].b = b_list[i].to('cpu')
        triples[i].c = c_list[i].to('cpu')

    return triples


class MatrixBeaverProvider(ParamProvider):
    def __init__(self, party=None):
        super().__init__(party=party)
        self.party = party
        self.param_type = MatrixTriples
        self.param = {}
        self.matrix_ptr = 0
        self.matrix_ptr_max = 0

    def load_mat_beaver(self):
        self.matrix_ptr_max = len(self.param)

    def load_param(self, x_shape=None, y_shape=None, num_of_party=2):
        file_name = f"MatrixBeaverTriples_{self.party.party_id}_{list(x_shape)}_{list(y_shape)}.pth"
        file_path = base_path + f"/aux_parameters/BeaverTriples/{num_of_party}party/Matrix"
        try:
            self.param = MatrixTriples.load_by_name(file_name, file_path)
        except FileNotFoundError:
            raise Exception("Need generate matrix triples in this shape first!")

    def get_parameters(self, x_shape=None, y_shape=None):
        if DEBUG_LEVEL == 2:
            if f"{list(x_shape)}_{list(y_shape)}" not in self.param.keys():
                a = torch.ones(x_shape, dtype=data_type, device=DEVICE)
                b = torch.ones(y_shape, dtype=data_type, device=DEVICE)

                assert x_shape[-1] == y_shape[-2]
                c_shape = torch.broadcast_shapes(x_shape[:-2], y_shape[:-2]) + (x_shape[-2], y_shape[-1])
                c = torch.ones(c_shape, dtype=data_type, device=DEVICE) * x_shape[-1]

                a_tensor = RingTensor(a, dtype=DTYPE).to(DEVICE)
                b_tensor = RingTensor(b, dtype=DTYPE).to(DEVICE)
                c_tensor = RingTensor(c, dtype=DTYPE).to(DEVICE)

                from crypto.primitives.arithmetic_secret_sharing.arithmetic_shared_ring_tensor import \
                    ArithmeticSharedRingTensor
                a = ArithmeticSharedRingTensor(a_tensor, self.party)
                b = ArithmeticSharedRingTensor(b_tensor, self.party)
                c = ArithmeticSharedRingTensor(c_tensor, self.party)

                self.param[f"{list(x_shape)}_{list(y_shape)}"] = MatrixTriples()
                self.param[f"{list(x_shape)}_{list(y_shape)}"].set_triples(a, b, c)
            return self.param[f"{list(x_shape)}_{list(y_shape)}"]
        else:
            mat_beaver = self.param[self.matrix_ptr].pop()
            self.matrix_ptr = (self.matrix_ptr + 1) % self.matrix_ptr_max
            return mat_beaver
