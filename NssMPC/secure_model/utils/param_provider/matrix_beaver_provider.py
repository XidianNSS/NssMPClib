#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

"""
Generate and load ASS and RSS matrix triples.
"""

import torch

from NssMPC.crypto.aux_parameter import MatmulTriples, RssMatmulTriples
from NssMPC.common.ring import RingTensor
from NssMPC.config import param_path, DEBUG_LEVEL, data_type, DEVICE, DTYPE
from NssMPC.secure_model.utils.param_provider.fixed_shape_provider import FixedShapeProvider


class MatrixBeaverProvider(FixedShapeProvider):
    """
    Matrix Beaver Provider for ArithmeticSecretSharing.
    """

    def __init__(self, party=None, saved_name=None, param_tag=None, root_path=None):
        """
        Initialize the MatrixBeaverProvider class.

        :param party: participant
        :type party: Party
        :param saved_name: Optional file name to save.
        :type saved_name: str
        :param param_tag: A label used to identify parameters
        :type param_tag: str
        :param root_path: The root path of the parameter file
        :type root_path: str
        """
        super().__init__(MatmulTriples, party, saved_name, param_tag, root_path)

    def load_param(self, x_shape=None, y_shape=None, num_of_party=2):
        """
        This method is responsible for loading the parameters for matrix.

        First, generate file names to load the corresponding matrix triples. Then try loading the saved triples from a specified path.

        :param x_shape: The shape of the matrix x
        :type x_shape: tuple
        :param y_shape: The shape of the matrix y
        :type y_shape: tuple
        :param num_of_party: the number of party
        :type num_of_party: int
        :raises FileNotFoundError: If the file cannot be found.
        """
        file_name = f"MatrixBeaverTriples_{self.party.party_id}_{list(x_shape)}_{list(y_shape)}.pth"
        file_path = param_path + f"BeaverTriples/{num_of_party}party/Matrix"
        try:
            self.param = MatmulTriples.load(file_path + file_name)
        except FileNotFoundError:
            raise Exception("Need generate matrix triples in this shape first!")

    def get_parameters(self, x_shape=None, y_shape=None):
        """
        Get the current matrix's parameters from the parameters.

        If DEBUG_LEVEL is 2:
            First check if there are already parameters for the shape. If not, generate a new parameter. The Beaver Triples are then created, creating tensors ``a`` and ``b`` with all 1, and calculating the shape of ``c`` based on the shape. Wrap these tensors with RingTensor to create the corresponding ASS object. Store these objects in the ``self.param`` dictionary.

        If DEBUG_LEVEL is not 2:
            Pop Beaver Triples from the argument list and update the pointer matrix_ptr

        :param x_shape: The shape of the matrix x
        :type x_shape: tuple
        :param y_shape: The shape of the matrix y
        :type y_shape: tuple
        :return: Beaver Triples
        :rtype: list
        """
        if DEBUG_LEVEL == 2:
            if f"{list(x_shape)}_{list(y_shape)}" not in self.param.keys():
                a = torch.ones(x_shape, dtype=data_type, device=DEVICE)
                b = torch.ones(y_shape, dtype=data_type, device=DEVICE)
                if len(x_shape) == 1 and len(y_shape) == 1:
                    x_shape = (1, x_shape[0])
                    y_shape = (y_shape[0], 1)
                elif len(x_shape) == 1:
                    x_shape = (1, x_shape[0])
                elif len(y_shape) == 1:
                    y_shape = (1, y_shape[0])
                assert x_shape[-1] == y_shape[-2]
                c_shape = torch.broadcast_shapes(x_shape[:-2], y_shape[:-2]) + (x_shape[-2], y_shape[-1])
                c = torch.ones(c_shape, dtype=data_type, device=DEVICE) * x_shape[-1] * 2  # todo: c != a @ b也对?

                a_tensor = RingTensor(a, dtype=DTYPE).to(DEVICE)
                b_tensor = RingTensor(b, dtype=DTYPE).to(DEVICE)
                c_tensor = RingTensor(c, dtype=DTYPE).to(DEVICE)

                from NssMPC.crypto.primitives.arithmetic_secret_sharing import \
                    ArithmeticSecretSharing
                a = ArithmeticSecretSharing(a_tensor)
                b = ArithmeticSecretSharing(b_tensor)
                c = ArithmeticSecretSharing(c_tensor)

                self.param[f"{list(x_shape)}_{list(y_shape)}"] = MatmulTriples()
                self.param[f"{list(x_shape)}_{list(y_shape)}"].set_triples(a, b, c)
            return self.param[f"{list(x_shape)}_{list(y_shape)}"]
        else:
            mat_beaver = self.param[self.matrix_ptr].pop()
            self.matrix_ptr = (self.matrix_ptr + 1) % self.matrix_ptr_max
            return mat_beaver.to(DEVICE)


class RssMatrixBeaverProvider(FixedShapeProvider):
    """
    Matrix Beaver Provider for ReplicatedSecretSharing.
    """

    def __init__(self, party=None, saved_name=None, param_tag=None, root_path=None):
        """
        Initialize the RssMatrixBeaverProvider class.

        :param party: participant
        :type party: Party
        :param saved_name: Optional file name to save.
        :type saved_name: str
        :param param_tag: A label used to identify parameters
        :type param_tag: str
        :param root_path: The root path of the parameter file
        :type root_path: str
        """
        super().__init__(RssMatmulTriples, party, saved_name, param_tag, root_path)

    def load_param(self, x_shape=None, y_shape=None, num_of_party=2):
        """
        This method is responsible for loading the parameters for matrix.

        First, generate file names to load the corresponding matrix triples. Then try loading the saved triples from a specified path.

        :param x_shape: The shape of the matrix x
        :type x_shape: tuple
        :param y_shape: The shape of the matrix y
        :type y_shape: tuple
        :param num_of_party: the number of party
        :type num_of_party: int
        :raises FileNotFoundError: If the file cannot be found.
        """
        file_name = f"MatrixBeaverTriples_{self.party.party_id}_{list(x_shape)}_{list(y_shape)}.pth"
        file_path = param_path + f"BeaverTriples/{num_of_party}party/Matrix"
        try:
            self.param = MatmulTriples.load(file_path + file_name)
        except FileNotFoundError:
            raise Exception("Need generate matrix triples in this shape first!")

    def get_parameters(self, x_shape=None, y_shape=None):
        """
        Get the current matrix's parameters from the parameters.

        If DEBUG_LEVEL is 2:
            First check if there are already parameters for the shape. If not, generate a new parameter. The Beaver Triples are then created, creating tensors ``a`` and ``b`` with all 1, and calculating the shape of ``c`` based on the shape. Wrap these tensors with RingTensor to create the corresponding RcSS object. Store these objects in the ``self.param`` dictionary.

        If DEBUG_LEVEL is not 2:
            Pop Beaver Triples from the argument list and update the pointer matrix_ptr

        :param x_shape: The shape of the matrix x
        :type x_shape: tuple
        :param y_shape: The shape of the matrix y
        :type y_shape: tuple
        :return: Beaver Triples
        :rtype: list
        """
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

                from NssMPC.crypto.primitives.arithmetic_secret_sharing import \
                    ReplicatedSecretSharing
                a = ReplicatedSecretSharing([a_tensor, a_tensor])
                b = ReplicatedSecretSharing([b_tensor, b_tensor])
                c = ReplicatedSecretSharing([c_tensor, c_tensor])

                self.param[f"{list(x_shape)}_{list(y_shape)}"] = RssMatmulTriples()
                self.param[f"{list(x_shape)}_{list(y_shape)}"].set_triples(a, b, c)
            return self.param[f"{list(x_shape)}_{list(y_shape)}"]
        else:
            mat_beaver = self.param[self.matrix_ptr].pop()
            self.matrix_ptr = (self.matrix_ptr + 1) % self.matrix_ptr_max
            return mat_beaver.to(DEVICE)
