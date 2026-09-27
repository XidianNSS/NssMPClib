import pytest
import torch

from nssmpc import HONEST_MAJORITY, Party3PC, PartyRuntime
from nssmpc.infra.tensor import RingTensor
from nssmpc.primitives.secret_sharing import ReplicatedSecretSharing as RSS
from nssmpc.protocols.honest_majority_3pc import comparison, msb_with_os
from nssmpc.protocols.honest_majority_3pc.mac_check import MAC_BUFFER


def share(values):
    return RSS.share(RingTensor.convert_to_ring(torch.as_tensor(values).float()))[0]


def test_comparison_does_not_mutate_operands(monkeypatch):
    x, y = share([1.0, -2.0]), share([2.0, -1.0])
    original = [s.item[i].tensor.clone() for s in (x, y) for i in (0, 1)]

    def fake_msb(z):
        return z * 0, z * 0, z * 0

    monkeypatch.setattr(msb_with_os, 'hm3pc_msb_with_os_without_mac_check', fake_msb)
    try:
        with PartyRuntime(Party3PC(0, HONEST_MAJORITY)):
            comparison.hm3pc_ge(x, y)
        assert x.dtype == y.dtype == 'float'
        for before, after in zip(original, [s.item[i].tensor for s in (x, y) for i in (0, 1)]):
            torch.testing.assert_close(before, after)
    finally:
        MAC_BUFFER.__init__()


@pytest.mark.parametrize('divisor', [0, 0.0, RingTensor.convert_to_ring(torch.tensor([1.0, 0.0]))])
def test_public_zero_rejected_before_communication(divisor):
    with pytest.raises(ZeroDivisionError):
        share([1.0, 2.0]) / divisor


@pytest.mark.parametrize('divisor', [float('nan'), float('inf')])
def test_public_nonfinite_rejected(divisor):
    with pytest.raises(ValueError, match='finite'):
        share([1.0, 2.0]) / divisor


def test_secret_division_rejects_incompatible_shapes_before_communication():
    with PartyRuntime(Party3PC(0, HONEST_MAJORITY)):
        with pytest.raises(RuntimeError):
            share(torch.ones(2, 3)) / share(torch.ones(4, 1))
