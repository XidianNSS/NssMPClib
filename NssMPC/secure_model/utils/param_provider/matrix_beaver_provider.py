import torch

from NssMPC.crypto.aux_parameter import MatmulTriples, RssMatmulTriples
from NssMPC.common.ring import RingTensor
from NssMPC.config import param_path, DEBUG_LEVEL, data_type, DEVICE, DTYPE
from NssMPC.secure_model.utils.param_provider.fixed_shape_provider import FixedShapeProvider


class MatrixBeaverProvider(FixedShapeProvider):
    def __init__(self, party=None, saved_name=None, param_tag=None, root_path=None):
        super().__init__(MatmulTriples, party, saved_name, param_tag, root_path)

    def load_param(self, x_shape=None, y_shape=None, num_of_party=2):
        file_name = f"MatrixBeaverTriples_{self.party.party_id}_{list(x_shape)}_{list(y_shape)}.pth"
        file_path = param_path + f"BeaverTriples/{num_of_party}party/Matrix"
        try:
            self.param = MatmulTriples.load(file_path + file_name)
        except FileNotFoundError:
            raise Exception("Need generate matrix triples in this shape first!")

    def get_parameters(self, x_shape=None, y_shape=None):
        if DEBUG_LEVEL == 2:
            if f"{list(x_shape)}_{list(y_shape)}" not in self.param.keys():
                a = torch.ones(x_shape, dtype=data_type, device=DEVICE)
                b = torch.ones(y_shape, dtype=data_type, device=DEVICE)

                assert x_shape[-1] == y_shape[-2]
                c_shape = torch.broadcast_shapes(x_shape[:-2], y_shape[:-2]) + (x_shape[-2], y_shape[-1])
                c = torch.ones(c_shape, dtype=data_type, device=DEVICE) * x_shape[-1] * 2  # todo: c != a @ b也对?

                a_tensor = RingTensor(a, dtype=DTYPE).to(DEVICE)
                b_tensor = RingTensor(b, dtype=DTYPE).to(DEVICE)
                c_tensor = RingTensor(c, dtype=DTYPE).to(DEVICE)

                from NssMPC.crypto.primitives.arithmetic_secret_sharing.arithmetic_secret_sharing import \
                    ArithmeticSecretSharing
                a = ArithmeticSecretSharing(a_tensor, self.party)
                b = ArithmeticSecretSharing(b_tensor, self.party)
                c = ArithmeticSecretSharing(c_tensor, self.party)

                self.param[f"{list(x_shape)}_{list(y_shape)}"] = MatmulTriples()
                self.param[f"{list(x_shape)}_{list(y_shape)}"].set_triples(a, b, c)
            return self.param[f"{list(x_shape)}_{list(y_shape)}"]
        else:
            mat_beaver = self.param[self.matrix_ptr].pop()
            self.matrix_ptr = (self.matrix_ptr + 1) % self.matrix_ptr_max
            return mat_beaver.to(DEVICE)


class RssMatrixBeaverProvider(FixedShapeProvider):
    def __init__(self, party=None, saved_name=None, param_tag=None, root_path=None):
        super().__init__(RssMatmulTriples, party, saved_name, param_tag, root_path)

    def load_param(self, x_shape=None, y_shape=None, num_of_party=2):
        file_name = f"MatrixBeaverTriples_{self.party.party_id}_{list(x_shape)}_{list(y_shape)}.pth"
        file_path = param_path + f"BeaverTriples/{num_of_party}party/Matrix"
        try:
            self.param = MatmulTriples.load(file_path + file_name)
        except FileNotFoundError:
            raise Exception("Need generate matrix triples in this shape first!")

    def get_parameters(self, x_shape=None, y_shape=None):
        if DEBUG_LEVEL == 2:
            if f"{list(x_shape)}_{list(y_shape)}" not in self.param.keys():
                a = torch.ones(x_shape, dtype=data_type, device=DEVICE)
                b = torch.ones(y_shape, dtype=data_type, device=DEVICE)

                assert x_shape[-1] == y_shape[-2]
                c_shape = torch.broadcast_shapes(x_shape[:-2], y_shape[:-2]) + (x_shape[-2], y_shape[-1])
                c = torch.ones(c_shape, dtype=data_type, device=DEVICE) * x_shape[-1] * 3

                a_tensor = RingTensor(a, dtype=DTYPE).to(DEVICE)
                b_tensor = RingTensor(b, dtype=DTYPE).to(DEVICE)
                c_tensor = RingTensor(c, dtype=DTYPE).to(DEVICE)

                from NssMPC.crypto.primitives.arithmetic_secret_sharing.replicated_secret_sharing import \
                    ReplicatedSecretSharing
                a = ReplicatedSecretSharing([a_tensor, a_tensor], self.party)
                b = ReplicatedSecretSharing([b_tensor, b_tensor], self.party)
                c = ReplicatedSecretSharing([c_tensor, c_tensor], self.party)

                self.param[f"{list(x_shape)}_{list(y_shape)}"] = RssMatmulTriples()
                self.param[f"{list(x_shape)}_{list(y_shape)}"].set_triples(a, b, c)
            return self.param[f"{list(x_shape)}_{list(y_shape)}"]
        else:
            mat_beaver = self.param[self.matrix_ptr].pop()
            self.matrix_ptr = (self.matrix_ptr + 1) % self.matrix_ptr_max
            return mat_beaver.to(DEVICE)
