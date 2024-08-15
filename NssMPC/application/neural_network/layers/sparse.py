import torch

from NssMPC.application.neural_network.functional.functional import torch2share


class SecEmbedding(torch.nn.Module):
    """
         onehot operation have been done in the offline phase and on the plaintext
         only support the embedding layer at the beginning of the network
    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0,
                 scale_grad_by_freq=False, sparse=False, _weight=None, _freeze=False, device=None, dtype=None):
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
        weight = torch2share(self.weight, x.__class__, x.dtype, x.party)
        z = x @ weight
        return z
