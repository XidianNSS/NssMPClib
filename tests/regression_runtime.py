"""Standalone real-process regression: python tests/regression_runtime.py DIR MODE.

MODE is prepare, 2pc, sh3pc, or hm3pc. Uses fresh test-only preprocessing in
DIR, free TCP ports, a 180 second cluster deadline and unconditional cleanup.
Run with DEVICE=cpu or DEVICE=cuda and the repository on PYTHONPATH.
"""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import torch

from nssmpc import HONEST_MAJORITY, SEMI_HONEST, Party2PC, Party3PC, PartyRuntime, SecretTensor
from nssmpc.config import DEVICE
from nssmpc.infra.tensor import RingTensor
from nssmpc.primitives.secret_sharing.function import SigmaDICFKey, VSigmaKey
from nssmpc.protocols.honest_majority_3pc.mac_check import MAC_BUFFER
from nssmpc.protocols.honest_majority_3pc.msb_with_os import MACKey
from nssmpc.protocols.honest_majority_3pc.multiplication import RssMatmulTriples, RssMulTriples
from nssmpc.protocols.honest_majority_3pc.truncation import RssTruncAuxParams
from nssmpc.protocols.semi_honest_2pc.b2a import B2AKey
from nssmpc.protocols.semi_honest_2pc.division import DivKey
from nssmpc.protocols.semi_honest_2pc.multiplication import AssMulTriples, MatmulTriples
from nssmpc.protocols.semi_honest_2pc.truncation import Wrap

torch.set_num_threads(1)

MATRICES = [
    ([1, 1], [1, 4]),
    ([1, 1], [1, 1]),
    ([2, 3, 4], [4, 2]),
    ([3, 4], [2, 4, 2]),
    ([2, 1, 3, 4], [1, 5, 4, 2]),
    ([1, 1, 3, 4], [1, 1, 4, 2]),
]


def parameter_dir(directory):
    path = str(Path(directory).resolve() / 'params') + '/'
    for name, module in list(sys.modules.items()):
        if name.startswith('nssmpc') and hasattr(module, 'param_path'):
            module.param_path = path


def prepare():
    count = int(os.environ.get('NSS_TEST_PARAM_COUNT', '512'))
    for cls in [AssMulTriples, Wrap, B2AKey, DivKey, SigmaDICFKey, RssMulTriples, RssTruncAuxParams, MACKey]:
        cls.gen_and_save(count)
    for tag in ['0', '1', '2']:
        VSigmaKey.gen_and_save(count, tag=tag)
        B2AKey.gen_and_save(count, tag=tag)
    for left, right in MATRICES:
        MatmulTriples.gen_and_save(2, left, right)
        RssMatmulTriples.gen_and_save(2, left, right)
    print('Prepared test-only parameters', flush=True)


def worker(directory, mode, rank, port):
    config = HONEST_MAJORITY if mode == 'hm3pc' else SEMI_HONEST
    cls = Party2PC if mode == '2pc' else Party3PC
    party = cls(rank, config, master_ip='127.0.0.1', master_port=str(port))
    results = []

    def secret(value):
        value = torch.as_tensor(value, dtype=torch.float32, device=DEVICE)
        return SecretTensor(tensor=value) if rank == 0 else SecretTensor(src_id=0)

    def check(label, result, expected, atol=0.04, rtol=0.02):
        if mode == 'hm3pc' and label.startswith('secret_div'):
            assert not MAC_BUFFER.x, 'division left comparison MACs unchecked'
        if mode == 'hm3pc' and MAC_BUFFER.x:
            MAC_BUFFER.check()
        actual = result.recon() if mode == '2pc' else result.recon(target_id=0)
        party.communicator.barrier()
        if rank == 0:
            actual = actual.convert_to_real_field().cpu()
            expected = torch.as_tensor(expected).cpu().to(actual.dtype)
            assert actual.shape == expected.shape, (label, actual.shape, expected.shape)
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
            error = (actual - expected).abs().max().item()
            results.append({'case': label, 'max_abs_error': error})
            print(label, 'PASS', error, flush=True)

    try:
        party.online()
        with PartyRuntime(party):
            for i, (left, right) in enumerate(MATRICES):
                a = (torch.arange(torch.Size(left).numel()).reshape(left) % 7).float() / 4
                b = (torch.arange(torch.Size(right).numel()).reshape(right) % 5).float() / 4
                check(f'matmul_{i}', secret(a) @ secret(b), a @ b)
            a = torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
            b = torch.tensor([[2.0], [-0.5]])
            check('multiply_broadcast', secret(a) * secret(b), a * b)
            for left_shape, right_shape in [
                ((), (2, 3)),
                ((2, 3), ()),
                ((3,), (2, 3)),
                ((2, 1), (1, 2)),
                ((2, 1, 3), (1, 4, 1)),
            ]:
                a = (torch.arange(torch.Size(left_shape).numel()).reshape(left_shape).float() + 1) / 4
                b = -(torch.arange(torch.Size(right_shape).numel()).reshape(right_shape).float() + 1) / 4
                check(f'multiply_broadcast_{left_shape}_{right_shape}', secret(a) * secret(b), a * b)
            for shape in [(), (1,), (2, 3)]:
                a = torch.full(shape, 1.5)
                b = torch.full(shape, -0.5)
                x, y = secret(a), secret(b)
                check(f'comparison_{shape}', x >= y, (a >= b).float(), atol=0, rtol=0)
                assert x.dtype == y.dtype == 'float', 'comparison mutated inputs'
                check(f'comparison_then_multiply_{shape}', x * y, a * b)
            a = torch.tensor([[-2.0, 0.0, 2.0], [-0.25, 1.0, 3.0]])
            b = torch.tensor([[0.0, 0.0, 1.0], [0.25, 2.0, 3.0]])
            check('comparison_mixed', secret(a) >= secret(b), (a >= b).float(), atol=0, rtol=0)
            if mode != '2pc':
                a = torch.tensor([[-3.0, 1.5, 4.0], [2.0, -4.0, 8.0]])
                for public in [2, -2, 0.5, -0.5, 1.5]:
                    check(f'public_div_{public}', secret(a) / public, a / public)
                b = torch.tensor([[0.5], [-2.0]])
                check('public_ring_div', secret(a) / RingTensor.convert_to_ring(b.to(DEVICE)), a / b)
            a = torch.tensor([1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 1.5, 3.0])
            b = torch.tensor([0.125, -0.25, 0.5, -1.0, 2.0, -4.0, 8.0, -16.0])
            check('secret_div_signed_range', secret(a) / secret(b), a / b, atol=0.06)
            a = torch.tensor([[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]])
            b = torch.tensor([[0.5], [-2.0]])
            check('secret_div_broadcast', secret(a) / secret(b), a / b, atol=0.06)
            check('secret_div_scalar', secret(3.0) / secret(-2.0), torch.tensor(-1.5), atol=0.06)
            b = torch.tensor([-7.25, -3.0, -0.7, 0.3, 1.3, 3.7, 7.25])
            a = torch.tensor([2.0, -3.0, 1.0, -0.5, 3.0, -4.0, 5.0])
            check('secret_div_non_powers', secret(a) / secret(b), a / b, atol=0.06)
        if rank == 0:
            Path(directory, f'{mode}_{DEVICE}.json').write_text(json.dumps(results, indent=2))
    finally:
        party.close()


def cluster(directory, mode):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    processes, logs = [], []
    try:
        for rank in range(2 if mode == '2pc' else 3):
            log = open(Path(directory, f'{mode}_{DEVICE}_p{rank}.log'), 'w')
            logs.append(log)
            processes.append(
                subprocess.Popen(
                    [sys.executable, __file__, directory, mode, str(rank), str(port)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
        deadline = time.monotonic() + 180
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError('Party failed; see logs in ' + directory)
            if time.monotonic() > deadline:
                raise TimeoutError('Cluster exceeded 180 seconds')
            time.sleep(0.2)
        assert all(p.returncode == 0 for p in processes)
        print(Path(directory, f'{mode}_{DEVICE}.json').read_text())
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait()
        for log in logs:
            log.close()


if __name__ == '__main__':
    directory, mode = sys.argv[1:3]
    Path(directory).mkdir(parents=True, exist_ok=True)
    parameter_dir(directory)
    if mode == 'prepare':
        prepare()
    elif len(sys.argv) == 5:
        worker(directory, mode, int(sys.argv[3]), int(sys.argv[4]))
    else:
        cluster(directory, mode)
