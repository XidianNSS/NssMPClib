from types import SimpleNamespace

import pytest
import torch

from nssmpc.infra.utils import cuda


@pytest.mark.parametrize(
    'left,right',
    [
        ((2, 3), (1, 3, 4)),
        ((1,), (1,)),
        ((2, 1), (1,)),
        ((2, 1, 3, 4), (1, 5, 4, 2)),
        ((3, 4), (2, 4, 2)),
        ((2, 3, 4), (4, 2)),
    ],
)
def test_cutlass_dispatch_preserves_matmul_shapes(monkeypatch, left, right):
    # Exercise the optional extension branch even on hosts without CUTLASS.
    monkeypatch.setattr(cuda, 'CUTLASS_AVAILABLE', True)
    monkeypatch.setattr(cuda, 'nss_cuda_ext', SimpleNamespace(fast_matmul_int32=torch.matmul), raising=False)
    monkeypatch.setattr(cuda, 'data_type', torch.int32)
    monkeypatch.setattr(cuda, 'RING_MAX', 2**32)
    x = torch.arange(torch.Size(left).numel(), dtype=torch.int32).reshape(left)
    y = torch.arange(torch.Size(right).numel(), dtype=torch.int32).reshape(right)
    torch.testing.assert_close(cuda.cuda_matmul(x, y), x @ y)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('dtype', [torch.int32, torch.int64])
def test_cuda_batched_ring_matmul_matches_cpu(monkeypatch, dtype):
    monkeypatch.setattr(cuda, 'CUTLASS_AVAILABLE', False)
    monkeypatch.setattr(cuda, 'data_type', dtype)
    monkeypatch.setattr(cuda, 'RING_MAX', 2 ** torch.iinfo(dtype).bits)
    x = torch.randint(-(2**30), 2**30, (2, 1, 3, 4), dtype=dtype)
    y = torch.randint(-(2**30), 2**30, (1, 5, 4, 2), dtype=dtype)
    torch.testing.assert_close(cuda.cuda_matmul(x.cuda(), y.cuda()).cpu(), x @ y)


@pytest.mark.skipif(not torch.cuda.is_available() or not cuda._HAS_TRITON, reason='requires CUDA and Triton')
def test_rotate_preserves_distinct_rows():
    x = torch.arange(4 * 513, device='cuda').reshape(4, 513)
    shifts = torch.tensor([0, 1, -2, 515], device='cuda')
    expected = torch.stack([torch.roll(row, shift) for row, shift in zip(x, [0, 1, -2, 515])])
    torch.testing.assert_close(cuda.cuda_rotate(x, shifts), expected)


@pytest.mark.skipif(not torch.cuda.is_available() or not cuda._HAS_TRITON, reason='requires CUDA and Triton')
@pytest.mark.parametrize('rows,shifts', [(1, [0, 1, -2]), (3, [2]), (3, [0, 1, -2])])
def test_rotate_broadcast_and_strided_input(rows, shifts):
    x = torch.arange(rows * 34, device='cuda').reshape(rows, 34)[:, ::2]
    s = torch.tensor(shifts, device='cuda')
    indices = (torch.arange(17, device='cuda') - s[:, None]) % 17
    expected = x[torch.arange(rows, device='cuda')[:, None], indices]
    torch.testing.assert_close(cuda.cuda_rotate(x, s), expected)
