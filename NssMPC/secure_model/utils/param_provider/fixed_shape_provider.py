#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

from NssMPC.crypto.aux_parameter import MatmulTriples
from NssMPC.config import param_path, DEBUG_LEVEL
from NssMPC.secure_model.utils.param_provider import BaseParamProvider


class FixedShapeProvider(BaseParamProvider):
    """
    Manage and provide specific shape parameter matrices (such as Beaver triples required for matrix multiplication)

    .. note:
        Since matrix operations are related to their shapes, we have set up the `FixedShapeProvider` class,
        where the special aspect is that the type of its `param` attribute is no longer the `Parameter` class, but a dictionary containing multiple `Parameter` class objects.
        The reason for this setup is that it is difficult to store auxiliary parameters of different shapes in a single `Parameter` object.
        Therefore, we change `param` to a dictionary and use the different shapes of the parameters as keys to distinguish between different `Parameter` objects.

    """

    def __init__(self, param_type, party=None, saved_name=None, param_tag=None, root_path=None):
        """
        ATTRIBUTES:
            * **party** (*Party*): participant
            * **param** (*dict*): Used to store parameters.
            * **matrix_ptr** (*int*): The pointer to the current matrix, used to track the current parameters in use.
            * **matrix_ptr_max** (*int*): The maximum matrix pointer

        """
        super().__init__(param_type, saved_name, param_tag, root_path)
        self.party = party
        self.param = {}
        self.matrix_ptr = 0
        self.matrix_ptr_max = 0

    def load_mat_beaver(self):
        """
        Load matrix information.

        Initialize the maximum value of the matrix pointer by setting ``matrix_ptr_max`` to the length of ``self.param``.
        This is usually called after the parameters are loaded to ensure that the pointer loops correctly.

        """
        self.matrix_ptr_max = len(self.param)

    def load_param(self, saved_name=None, *shapes, num_of_party=2):
        """
        This method is responsible for loading the parameters of the specified shape.

        If no ``saved_name`` is provided, the file name is generated based on the MatmulTriples class name, shape,
        and participant ID. Otherwise, use the provided ``saved_name`` to generate the file name. Then load the file in
        the specified directory to ``self.param``.

        :param saved_name: Optional file name to save.
        :type saved_name: str
        :param shapes: The shapes of a matrix
        :type shapes: torch.Tensor
        :param num_of_party: the number of parties
        :type num_of_party: int
        :raises FileNotFoundError: If the file cannot be found.
        """
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
        """
        Get the current matrix's parameters from the parameters.

        If debugging 1 or 2, return the first element of the current parameter.

        If debugging 0 or non-debugging, pop the last element of the current parameter. The matrix pointer is then updated.

        :param shapes: The shapes of a matrix
        :type shapes: torch.Tensor
        :return: Obtained parameters
        :rtype: torch.Tensor
        """
        if DEBUG_LEVEL:
            res = self.param[self.matrix_ptr][0]
        else:
            res = self.param[self.matrix_ptr].pop()
        self.matrix_ptr = (self.matrix_ptr + 1) % self.matrix_ptr_max
        return res
