"""
SecEmbedding is a custom embedding layer that extends PyTorch's nn.Module.

It is primarily used for handling one-hot encoded inputs and is used in secure computing environments
(e.g., privacy-preserving machine learning). The design of this layer makes it possible to safely share
the embedding weights in multi-party computation.

Onehot operation have been done in the offline phase and on the plaintext only support the embedding layer
at the beginning of the communication.
"""
#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import torch

from nssmpc.application.neural_network.functional.functional import torch2share


class SecEmbedding(torch.nn.Module):
    """
    SecEmbedding is a custom embedding layer that extends PyTorch's nn.Module.

    It is primarily used for handling one-hot encoded inputs and is used in secure computing environments
    (e.g., privacy-preserving machine learning). The design of this layer makes it possible to safely share
    the embedding weights in multi-party computation.

    Onehot operation have been done in the offline phase and on the plaintext only support the embedding layer
    at the beginning of the communication.
    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0,
                 scale_grad_by_freq=False, sparse=False, _weight=None, _freeze=False, device=None, dtype=None):
        """
        Initializes the SecEmbedding layer.

        Args:
            num_embeddings (int): The size of the embedded dictionary, the number of different identifiers that can be embedded.
            embedding_dim (int): Each embedded dimension represents the size of the vector space that each identifier will be mapped to.
            padding_idx (int, optional): Specified fill index.
            max_norm (float, optional): The maximum norm constraint to the embedded vector.
            norm_type (float, optional): The type of norm (default is 2.0), indicates the use of the L2 norm.
            scale_grad_by_freq (bool, optional): The gradient update will be scaled according to the frequency.
            sparse (bool, optional): Whether to use sparse updates.
            _weight (torch.Tensor, optional): Optional initial embedding weights.
            _freeze (bool, optional): Control whether the embedded weights are trainable (defaulting to False).
            device (str, optional): The device where tensors are stored.
            dtype (str, optional): Data type.

        Examples:
            >>> layer = SecEmbedding(num_embeddings=10, embedding_dim=3)
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
        Performs the forward pass of the embedding layer.

        First, converts the filled input into a column format. Then, calculates the dot product between the input
        and the shared weight to obtain the embedded output.

        Args:
            x (ArithmeticSecretSharing or ReplicatedSecretSharing): Input tensor.

        Returns:
            ArithmeticSecretSharing or ReplicatedSecretSharing: A tensor that contains an embedded vector.

        Examples:
            >>> output = layer(input_tensor)
        """
        weight = torch2share(self.weight, x.__class__, x.dtype)
        z = x @ weight
        return z
