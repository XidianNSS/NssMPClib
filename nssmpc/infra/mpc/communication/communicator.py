from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class Communicator(ABC):
    """Abstract communicator interface defining basic distributed communication operations.

    This interface provides a unified abstraction across different communication backends
    (e.g., TensorPipe, gRPC, MPI).
    """

    @abstractmethod
    def __init__(self, rank: int, world_size: int, **kwargs):
        """Initialize the communicator.

        Args:
            rank: Rank of the current process (0 to world_size-1)
            world_size: Total number of processes
            **kwargs: Backend-specific configuration parameters
        """
        pass

    @abstractmethod
    def send(self, target_rank: int, item: Any) -> Optional[Any]:
        """Send an item to the target process.

        Args:
            target_rank: Rank of the target process
            item: Item to send (e.g., a tensor)

        Returns:
            None or a handle for the send operation (e.g., a Future object). The exact type is implementation-dependent.
        """
        pass

    @abstractmethod
    def recv(self, source_rank: int) -> Any:
        """Receive an item from the source process.

        Args:
            source_rank: Rank of the source process

        Returns:
            The received item (e.g., a tensor)
        """
        pass

    @abstractmethod
    def barrier(self):
        """Barrier synchronization: synchronize all processes at this point.

        This method blocks until all processes have reached this barrier.
        """
        pass

    @abstractmethod
    def wait_for_peers(self, timeout: float = 60.0):
        """Wait for all peer nodes to be ready.

        Args:
            timeout: Timeout in seconds
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get communication statistics.

        Returns:
            Dictionary containing statistics information
        """
        pass

    @abstractmethod
    def shutdown(self):
        """Shut down the communicator and release resources."""
        pass
