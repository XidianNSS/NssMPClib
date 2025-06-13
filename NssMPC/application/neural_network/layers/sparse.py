"""
SecEmbedding is a custom embedding layer that extends PyTorch's nn.Module. It is primarily used for handling
one-hot encoded inputs and is used in secure computing environments (e.g., privacy-preserving machine learning). The
design of this layer makes it possible to safely share the embedding weights in multi-party computation.
"""
#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import torch
from NssMPC.application.neural_network.functional.functional import torch2share


class SecEmbedding(torch.nn.Module):
    """
    onehot operation have been done in the offline phase and on the plaintext only support the embedding layer at the beginning of the network.
    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0,
                 scale_grad_by_freq=False, sparse=False, _weight=None, _freeze=False, device=None, dtype=None):
        """
        :param num_embeddings: The size of the embedded dictionary, the number of different identifiers that can be embedded.
        :type num_embeddings: int
        :param embedding_dim: Each embedded dimension represents the size of the vector space that each identifier will be mapped to.
        :type embedding_dim: int
        :param padding_idx: Specified fill index
        :type padding_idx: int
        :param max_norm: the maximum norm constraint to the embedded vector.
        :type max_norm: int
        :param norm_type: The type of norm (default is **2.0**), indicates the use of the L2 norm.
        :type norm_type: float
        :param scale_grad_by_freq: The gradient update will be scaled according to the frequency.
        :type scale_grad_by_freq: bool
        :param sparse: Whether to use sparse updates.
        :type sparse: bool
        :param _weight: Optional initial embedding weights.
        :type _weight: torch.Tensor
        :param _freeze: Control whether the embedded weights are trainable (defaulting to **False**).
        :type _freeze: torch.Tensor
        :param device: The device where tensors are stored.
        :type device: str
        :param dtype: data type
        :type dtype: str
        """
        super(SecEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        if _weight is None:
            self.weight = torch.nn.Parameter(
                torch.zeros([num_embeddings, embedding_dim], dtype=torch.int64), requires_grad=False)
        else:
            assert list(_weight.shape) == [num_embeddings, embedding_dim], \
                'Shape of weight does not match num_embeddings and embedding_dim'
            self.weight = torch.nn.Parameter(_weight, requires_grad=False)

        self.sparse = sparse

    def forward(self, x):
        """
        First, use :func:`~NssMPC.application.neural_network.functional.functional.torch2share` to convert the filled input
        into a column format. Then, Using the **@** operator for matrix multiplication, calculate the dot product between the input ``x`` and the shared weight weight to obtain the embedded output ``z``.

        :param x: Input tensor
        :type x: ArithmeticSecretSharing or ReplicatedSecretSharing
        :return: A tensor that contains an embedded vector.
        :rtype: ArithmeticSecretSharing or ReplicatedSecretSharing
        """
        weight = torch2share(self.weight, x.__class__, x.dtype)
        z = x @ weight
        return z
