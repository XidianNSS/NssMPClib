import threading
import time
import uuid
from collections import defaultdict, deque
from typing import Any, Dict

import torch
import torch.distributed.rpc as rpc

from nssmpc.config import DEVICE
from nssmpc.infra.mpc.communication.communicator import Communicator
from nssmpc.infra.utils.profiling import count_bytes, bytes_convert

# 全局消息存储（每个rank独立）
_message_queues: Dict[str, deque] = defaultdict(deque)
_message_storage: Dict[str, Any] = {}
_queue_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
_condition_vars: Dict[str, threading.Condition] = defaultdict(threading.Condition)


def _store_message(tensor: torch.Tensor, key: str, source_rank: int) -> bool:
    """Stores a message in the target rank.

    Args:
        tensor: The tensor data to store.
        key: The unique key for the message.
        source_rank: The rank of the source worker.

    Returns:
        bool: True if stored successfully.

    Examples:
        >>> _store_message(torch.tensor([1]), "msg_1", 0)
    """
    queue_key = f"{source_rank}"

    with _queue_locks[queue_key]:
        _message_storage[key] = tensor
        _message_queues[queue_key].append(key)

        if queue_key in _condition_vars:
            with _condition_vars[queue_key]:
                _condition_vars[queue_key].notify_all()

    return True


def _retrieve_message(source_rank: int, timeout: float = 60.0) -> torch.Tensor:
    """Retrieves a message from the current rank with a waiting mechanism.

    Args:
        source_rank: The rank of the source worker.
        timeout: The maximum time to wait for the message in seconds.

    Returns:
        torch.Tensor: The retrieved tensor.

    Raises:
        TimeoutError: If the message is not received within the timeout.

    Examples:
        >>> tensor = _retrieve_message(0, timeout=10.0)
    """
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
    """Simple ping function to check connection.

    Returns:
        str: "pong" string.

    Examples:
        >>> response = _ping()
    """
    return "pong"


class TensorPipeCommunicator(Communicator):
    """Handles tensor communication using PyTorch RPC and TensorPipe.

    Attributes:
        rank (int): The rank of the current worker.
        world_size (int): The total number of workers.
        device (str): The device to use (e.g., 'cpu', 'cuda').
        comm_stats (dict): Statistics about communication.
    """

    def __init__(self, rank: int, world_size: int, master_addr: str = "localhost", master_port: str = "29500"):
        """Initializes the TensorPipeCommunicator.

        Args:
            rank: The rank of the current worker.
            world_size: The total number of workers.
            master_addr: The address of the master node.
            master_port: The port of the master node.

        Examples:
            >>> comm = TensorPipeCommunicator(0, 2)
        """
        self.rank = rank
        self.world_size = world_size
        self.device = DEVICE

        print(f"Rank {rank}: Setting up RPC with master {master_addr}:{master_port}")

        options = rpc.TensorPipeRpcBackendOptions(
            num_worker_threads=1,
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

        self._barrier_counter = 0

    def wait_for_peers(self, timeout: float = 60.0):
        """Waits for all peer nodes to be ready.

        Args:
            timeout: The maximum time to wait in seconds.

        Raises:
            TimeoutError: If a peer does not become ready within the timeout.

        Examples:
            >>> comm.wait_for_peers()
        """
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

    def send(self, target_rank: int, item: Any):
        """Sends an object to the target rank.

        Args:
            target_rank: The rank of the destination worker.
            item: The item to send.

        Returns:
            torch.distributed.rpc.Future: A future object representing the asynchronous operation.

        Raises:
            ValueError: If sending to self.

        Examples:
            >>> fut = comm.send(1, torch.tensor([1, 2, 3]))
        """
        if target_rank == self.rank:
            raise ValueError("Cannot send to self")
        message_id = str(uuid.uuid4())

        fut = rpc.rpc_async(
            f"worker{target_rank}",
            _store_message,
            args=(item, message_id, int(rpc.get_worker_info().name.replace('worker', '')))
        )

        self.comm_stats["send_count"] += 1
        self.comm_stats["send_bytes"] += count_bytes(item)

        return fut

    def recv(self, source_rank: int, timeout: float = 60.0) -> torch.Tensor:
        """Receives a tensor from the source rank.

        Args:
            source_rank: The rank of the source worker.
            timeout: The maximum time to wait in seconds.

        Returns:
            torch.Tensor: The received tensor.

        Raises:
            ValueError: If receiving from self.

        Examples:
            >>> tensor = comm.recv(0)
        """
        if source_rank == self.rank:
            raise ValueError("Cannot receive from self")

        # print(f"Rank {self.rank}: Requesting tensor from rank {source_rank}")

        tensor = _retrieve_message(source_rank, timeout)
        if DEVICE != "cpu":
            torch.cuda.synchronize()

        # print(f"Rank {self.rank}: Received tensor {tensor.shape} from rank {source_rank}")
        self.comm_stats["recv_count"] += 1
        self.comm_stats["recv_bytes"] += count_bytes(tensor)

        return tensor

    def barrier(self):
        """Performs barrier synchronization across all workers.

        Examples:
            >>> comm.barrier()
        """
        if self.world_size == 1:
            return

        print(f"Rank {self.rank}: Barrier synchronization")

        for target_rank in range(self.world_size):
            if target_rank != self.rank:
                fut = self.send(target_rank, 42)

        for source_rank in range(self.world_size):
            if source_rank != self.rank:
                signal = self.recv(source_rank)

        print(f"Rank {self.rank}: Barrier completed")

    def get_stats(self):
        """Gets communication statistics.

        Returns:
            dict: A dictionary containing send/recv counts, bytes, and elapsed time.

        Examples:
            >>> stats = comm.get_stats()
        """
        elapsed = time.time() - self.comm_stats["start_time"]
        stats = self.comm_stats.copy()
        stats["elapsed_time"] = elapsed
        stats["device"] = str(self.device)
        return stats

    def shutdown(self):
        """Shuts down the communicator and RPC backend.

        Examples:
            >>> comm.shutdown()
        """
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
