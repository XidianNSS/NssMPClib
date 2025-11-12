from datetime import datetime

import torch
import torch.distributed.rpc as rpc
from typing import Any, Dict
import time
import threading
from collections import defaultdict, deque
import uuid

from NssMPC.common.utils import count_bytes, bytes_convert
from NssMPC.config import DEVICE

# 全局消息存储（每个rank独立）
_message_queues: Dict[str, deque] = defaultdict(deque)
_message_storage: Dict[str, Any] = {}
_queue_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
_condition_vars: Dict[str, threading.Condition] = defaultdict(threading.Condition)


def _store_message(tensor: torch.Tensor, key: str, source_rank:int) -> bool:
    """在目标rank存储消息"""
    # 使用统一的队列键
    queue_key = f"{source_rank}"

    with _queue_locks[queue_key]:
        _message_storage[key] = tensor
        _message_queues[queue_key].append(key)

        # 通知等待的接收方
        if queue_key in _condition_vars:
            with _condition_vars[queue_key]:
                _condition_vars[queue_key].notify_all()

    # current_rank = int(rpc.get_worker_info().name.replace('worker', ''))
    # print(f"Rank {current_rank}: Stored message, key: {key}")
    return True


def _retrieve_message(source_rank:int, timeout: float = 60.0) -> torch.Tensor:
    """从当前rank获取消息（带等待机制）"""
    queue_key = f"{source_rank}"
    current_rank = int(rpc.get_worker_info().name.replace('worker', ''))

    # print(f"Rank {current_rank}: Waiting for message")

    start_time = time.time()

    while time.time() - start_time < timeout:
        with _queue_locks[queue_key]:
            if _message_queues[queue_key]:
                key = _message_queues[queue_key].popleft()
                tensor = _message_storage.pop(key, None)
                if tensor is not None:
                    # print(f"Rank {current_rank}: Retrieved message, key: {key}")
                    return tensor

        # 等待通知或超时
        if queue_key not in _condition_vars:
            _condition_vars[queue_key] = threading.Condition()

        with _condition_vars[queue_key]:
            remaining_time = timeout - (time.time() - start_time)
            if remaining_time > 0:
                _condition_vars[queue_key].wait(timeout=min(2.0, remaining_time))
            else:
                break

    raise TimeoutError(f"Rank {current_rank}: Timeout waiting for message")


def _ping() -> str:
    return "pong"


class TensorPipeCommunicator:
    def __init__(self, rank: int, world_size: int, master_addr: str = "localhost", master_port: str = "29500"):
        self.rank = rank
        self.world_size = world_size
        self.device = DEVICE

        print(f"Rank {rank}: Setting up RPC with master {master_addr}:{master_port}")

        options = rpc.TensorPipeRpcBackendOptions(
            num_worker_threads=16,
            rpc_timeout=120,
            init_method=f"tcp://{master_addr}:{master_port}",
            _transports=["uv"]
        )

        for i in range(world_size):
            if i != rank:
                options.set_device_map(f"worker{i}", {0: 0})

        rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )

        self.comm_stats = {
            "send_count": 0,
            "recv_count": 0,
            "send_bytes": 0,
            "recv_bytes": 0,
            "start_time": time.time()
        }

        # 屏障计数器
        self._barrier_counter = 0

    def wait_for_peers(self, timeout: float = 60.0):
        """等待所有对等节点就绪"""
        print(f"Rank {self.rank}: Waiting for all peers to be ready...")
        start_time = time.time()

        for target_rank in range(self.world_size):
            if target_rank != self.rank:
                connected = False
                while time.time() - start_time < timeout and not connected:
                    try:
                        result = rpc.rpc_sync(f"worker{target_rank}", _ping, timeout=5.0)
                        if result == "pong":
                            connected = True
                            print(f"Rank {self.rank}: Connected to rank {target_rank}")
                    except Exception as e:
                        if "timeout" in str(e).lower():
                            time.sleep(2)
                        else:
                            time.sleep(1)

                if not connected:
                    raise TimeoutError(f"Rank {self.rank}: Failed to connect to rank {target_rank}")

        print(f"Rank {self.rank}: All peers are ready!")

    def send(self, target_rank: int, tensor: torch.Tensor):
        """发送张量到目标rank"""
        if target_rank == self.rank:
            raise ValueError("Cannot send to self")
        # if hasattr(tensor,"contiguous"):
        #     tensor = tensor.to(self.device).contiguous()
        message_id = str(uuid.uuid4())
        # print(f"Rank {self.rank}: Sending tensor {tensor.shape} to rank {target_rank}")

        fut = rpc.rpc_async(
            f"worker{target_rank}",
            _store_message,
            args=(tensor, message_id, int(rpc.get_worker_info().name.replace('worker', '')))
        )

        self.comm_stats["send_count"] += 1
        self.comm_stats["send_bytes"] += count_bytes(tensor)

        return fut

    def recv(self, source_rank: int, timeout: float = 60.0) -> torch.Tensor:
        """从源rank接收张量"""
        if source_rank == self.rank:
            raise ValueError("Cannot receive from self")

        # print(f"Rank {self.rank}: Requesting tensor from rank {source_rank}")

        tensor = _retrieve_message(source_rank,timeout)
        if DEVICE != "cpu":
            torch.cuda.synchronize()

        # print(f"Rank {self.rank}: Received tensor {tensor.shape} from rank {source_rank}")
        self.comm_stats["recv_count"] += 1
        self.comm_stats["recv_bytes"] += count_bytes(tensor)

        return tensor


    def barrier(self):
        """屏障同步"""
        if self.world_size == 1:
            return

        print(f"Rank {self.rank}: Barrier synchronization")

        for target_rank in range(self.world_size):
            if target_rank != self.rank:
                fut = self.send(target_rank,42)

        for source_rank in range(self.world_size):
            if source_rank != self.rank:
                signal = self.recv(source_rank)

        print(f"Rank {self.rank}: Barrier completed")

    def get_stats(self):
        """获取通信统计"""
        elapsed = time.time() - self.comm_stats["start_time"]
        stats = self.comm_stats.copy()
        stats["elapsed_time"] = elapsed
        stats["device"] = str(self.device)
        return stats

    def shutdown(self):
        """关闭通信器"""
        try:
            print(f"Rank {self.rank}: Shutting down RPC")
            stats = self.get_stats()
            print(f"Rank {self.rank} Communication stats:")
            print(f"  Send operations: {stats['send_count']}")
            print(f"  Receive operations: {stats['recv_count']}")
            print(f"  Send bytes: {bytes_convert(stats['send_bytes'])}")
            print(f"  Receive bytes: {bytes_convert(stats['recv_bytes'])}")
            print(f"  Elapsed time: {stats['elapsed_time']:.2f} seconds")

            # 清理存储
            global _message_queues, _message_storage
            _message_queues.clear()
            _message_storage.clear()

            rpc.shutdown()
            print(f"Rank {self.rank}: Shutdown completed")
        except Exception as e:
            print(f"Rank {self.rank}: Warning during shutdown - {e}")