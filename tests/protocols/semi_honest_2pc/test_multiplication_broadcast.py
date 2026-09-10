import pytest
import torch

from nssmpc.infra.tensor import RingTensor
from nssmpc.primitives.secret_sharing import AdditiveSecretSharing
from nssmpc.protocols.semi_honest_2pc import multiplication


class FakeParty:
    party_id = 1

    def get_param(self, param_type, num):
        zeros = RingTensor.zeros([num])
        share = AdditiveSecretSharing(zeros)
        return share, share, share


@pytest.mark.parametrize(
    'left_shape,right_shape',
    [
        ((2, 3), (2, 1)),
        ((), (2, 3)),
        ((2, 3), ()),
        ((3,), (2, 3)),
        ((2, 1), (1, 2)),
        ((2, 1, 3), (1, 4, 1)),
    ],
)
def test_sh2pc_mul_beaver_broadcasts_operands(monkeypatch, left_shape, right_shape):
    left_value = torch.arange(torch.Size(left_shape).numel()).reshape(left_shape).float() + 1
    right_value = -torch.arange(torch.Size(right_shape).numel()).reshape(right_shape).float() - 1
    left_plain = RingTensor.convert_to_ring(left_value)
    right_plain = RingTensor.convert_to_ring(right_value)
    left = AdditiveSecretSharing(left_plain)
    right = AdditiveSecretSharing(right_plain)

    monkeypatch.setattr(
        AdditiveSecretSharing,
        "recon",
        lambda self: self.item,
    )

    result = multiplication.sh2pc_mul_beaver(left, right, FakeParty())

    assert result.shape == (left_value * right_value).shape
    expected = RingTensor.mul(left_plain, right_plain)
    assert torch.equal(result.item.tensor, expected.tensor)


def test_sh2pc_mul_beaver_rejects_incompatible_shapes():
    left = AdditiveSecretSharing(RingTensor.zeros([2, 3]))
    right = AdditiveSecretSharing(RingTensor.zeros([4, 1]))

    with pytest.raises(ValueError, match="could not be broadcast together"):
        multiplication.sh2pc_mul_beaver(left, right, FakeParty())
