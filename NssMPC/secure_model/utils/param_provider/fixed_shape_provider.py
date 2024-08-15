from NssMPC.crypto.aux_parameter import MatmulTriples
from NssMPC.config import param_path, DEBUG_LEVEL
from NssMPC.secure_model.utils.param_provider._base_param_provider import BaseParamProvider


class FixedShapeProvider(BaseParamProvider):
    def __init__(self, param_type, party=None, saved_name=None, param_tag=None, root_path=None):
        super().__init__(param_type, saved_name, param_tag, root_path)
        self.party = party
        self.param = {}
        self.matrix_ptr = 0
        self.matrix_ptr_max = 0

    def load_mat_beaver(self):
        self.matrix_ptr_max = len(self.param)

    def load_param(self, saved_name=None, *shapes, num_of_party=2):
        shapes_str = '_'.join([str(shape) for shape in shapes])
        if saved_name is None:
            file_name = f"{MatmulTriples.__name__}_{shapes_str}_{self.party.party_id}.pth"
            file_path = param_path + f"{MatmulTriples.__name__}/"

        else:
            file_name = f"{saved_name}_{shapes_str}_{self.party.party_id}.pth"
            file_path = param_path + f"{saved_name}/"
        try:
            self.param = MatmulTriples.load(file_path + file_name)
        except FileNotFoundError:
            raise Exception("Need generate matrix triples in this shape first!")

    def get_parameters(self, *shapes):
        if DEBUG_LEVEL:
            res = self.param[self.matrix_ptr][0]
        else:
            res = self.param[self.matrix_ptr].pop()
        self.matrix_ptr = (self.matrix_ptr + 1) % self.matrix_ptr_max
        return res
