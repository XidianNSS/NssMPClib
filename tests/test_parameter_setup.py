"""Run the original matrix and NN test entry points with fresh matrix parameters.

Run with DEVICE=cpu and the repository on PYTHONPATH. CUDA cases are skipped
when unavailable. No pretrained weights or external datasets are required.
"""

import importlib
import os
import runpy
import shutil
import socket
import subprocess
import sys
import time
import unittest
from pathlib import Path

import pytest
import torch

from tests.regression_runtime import (
    AssMulTriples,
    B2AKey,
    MACKey,
    RssMulTriples,
    RssTruncAuxParams,
    SigmaDICFKey,
    VSigmaKey,
    Wrap,
    parameter_dir,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def auxiliary_parameters(tmp_path_factory):
    directory = tmp_path_factory.mktemp('test_auxiliary_parameters')
    previous = {
        module: module.param_path
        for name, module in list(sys.modules.items())
        if name.startswith('nssmpc') and hasattr(module, 'param_path')
    }
    parameter_dir(directory)
    try:
        for cls in [AssMulTriples, Wrap, B2AKey, SigmaDICFKey, RssMulTriples, RssTruncAuxParams, MACKey]:
            cls.gen_and_save(512)
        for tag in ['0', '1', '2']:
            VSigmaKey.gen_and_save(512, tag=tag)
            B2AKey.gen_and_save(512, tag=tag)
    finally:
        for module, path in previous.items():
            module.param_path = path
    return directory


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('mode', ['matrix', 'nn2pc', 'nn3pc'])
def test_original_entry_points_prepare_matrix_triples(tmp_path, auxiliary_parameters, device, mode):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('requires CUDA')
    shutil.copytree(auxiliary_parameters / 'params', tmp_path / 'params')
    triple_type = 'RssMatmulTriples' if mode == 'nn3pc' else 'MatmulTriples'
    assert not (tmp_path / 'params' / triple_type).exists()

    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    processes, logs = [], []
    environment = dict(os.environ, DEVICE=device, MASTER_PORT=str(port), PYTHONPATH=str(ROOT))
    try:
        for rank in range(3 if mode == 'nn3pc' else 2):
            log = (tmp_path / f'p{rank}.log').open('w')
            logs.append(log)
            processes.append(
                subprocess.Popen(
                    [sys.executable, __file__, '--worker', mode, str(rank), str(tmp_path)],
                    cwd=ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
        deadline = time.monotonic() + 120
        while any(process.poll() is None for process in processes):
            if any(process.poll() not in (None, 0) for process in processes):
                break
            if time.monotonic() >= deadline:
                pytest.fail(f'{mode} exceeded 120 seconds; logs: {tmp_path}')
            time.sleep(0.1)
        failed = [process.returncode for process in processes if process.poll() != 0]
        output = '\n'.join((tmp_path / f'p{rank}.log').read_text() for rank in range(len(processes)))
        assert not failed, output
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait()
        for log in logs:
            log.close()

    shapes = '[10, 10]_[10, 10]' if mode == 'matrix' else '[1, 2]_[2, 3]'
    for rank in range(len(processes)):
        assert (tmp_path / 'params' / triple_type / f'p{rank}_{triple_type}_{shapes}.pth').is_file()


def run_worker(mode, rank, directory):
    # Load the model helper before overriding the imported parameter paths.
    import tests.parameter_setup  # noqa: F401

    parameter_dir(directory)
    if mode == 'matrix':
        module = importlib.import_module(f'tests.primitives.secret_sharing.test_ass_p{rank}')
        case = getattr(module, f'TestParty{rank}')('test_matrix_multiplication')
        result = unittest.TextTestRunner().run(unittest.TestSuite([case]))
        if not result.wasSuccessful():
            raise SystemExit(1)
    else:
        relative = (
            f'tests/application/neural_network/3pc/P{rank}.py'
            if mode == 'nn3pc'
            else f'tests/application/neural_network/2pc/neural_network_P{rank}.py'
        )
        path = ROOT / relative
        sys.argv = [str(path), '--smoke']
        runpy.run_path(str(path), run_name='__main__')


if __name__ == '__main__':
    if len(sys.argv) != 5 or sys.argv[1] != '--worker':
        raise SystemExit('Run this module with pytest.')
    run_worker(sys.argv[2], int(sys.argv[3]), sys.argv[4])
