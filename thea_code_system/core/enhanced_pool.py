#!/usr/bin/env python
"""
Enhanced Actor Pool
Wraps Ray's ActorPool with our PyTorch scalar operations and MPS optimizations
"""

import ray
from ray.util.actor_pool import ActorPool as RayActorPool
from ray.util.queue import Queue as RayQueue
import torch
import asyncio
from typing import Any, Callable, List, Optional, Dict, TypeVar
from dataclasses import dataclass
import time

from .base import BaseActor, ActorConfig, WorkerMetrics
from .scalars import ScalarOperations
from .mps import MPSDevice


T = TypeVar('T')


@dataclass
class PoolMetrics:
    """Enhanced metrics for pool operations"""
    total_tasks_submitted: int = 0
    total_tasks_completed: int = 0
    total_gpu_operations: int = 0
    total_tensor_bytes: int = 0
    average_latency_ms: float = 0.0
    peak_memory_mb: float = 0.0
    

class EnhancedActor(BaseActor):
    """
    Enhanced actor that combines Ray's capabilities with PyTorch operations
    """
    
    def __init__(self, actor_id: str, config: Optional[ActorConfig] = None):
        super().__init__(actor_id, config)
        self.scalar_ops = None
        self.mps_device = None
        self._tensor_cache = {}
        
    async def initialize(self) -> None:
        """Initialize with PyTorch and MPS support"""
        # Setup PyTorch scalar operations
        self.scalar_ops = ScalarOperations(self.device)
        
        # Setup MPS if available
        if self.device.type == "mps":
            self.mps_device = MPSDevice()
            
        self._initialized = True
        self.logger.info(f"Enhanced actor {self.actor_id} initialized on {self.device}")
    
    async def process_with_torch(self, data: Any) -> Any:
        """Process data ensuring all math uses PyTorch"""
        start = time.perf_counter()
        
        # Convert to tensor if numeric
        if isinstance(data, (int, float)):
            tensor = self.scalar_ops.scalar(data)
        else:
            tensor = data
            
        # Track metrics
        self.metrics.items_processed += 1
        self.metrics.total_processing_time += (time.perf_counter() - start)
        
        return tensor
    
    async def share_tensor(self, key: str, tensor: torch.Tensor) -> None:
        """Store tensor for sharing with other actors"""
        self._tensor_cache[key] = tensor
        
    async def get_shared_tensor(self, key: str) -> Optional[torch.Tensor]:
        """Retrieve shared tensor"""
        return self._tensor_cache.get(key)
    
    def clear_tensor_cache(self) -> None:
        """Clear tensor cache to free memory"""
        self._tensor_cache.clear()
        if self.mps_device:
            self.mps_device.clear_cache()


class EnhancedActorPool:
    """
    Enhanced actor pool that wraps Ray's ActorPool with additional features:
    - All mathematical operations use PyTorch scalars
    - MPS optimization for Apple Silicon
    - Tensor sharing between actors
    - Advanced metrics and monitoring
    """
    
    def __init__(
        self,
        actor_class: type,
        num_actors: int = 4,
        config: Optional[ActorConfig] = None,
        enable_queue: bool = False,
        queue_size: int = 100
    ):
        self.actor_class = actor_class
        self.num_actors = num_actors
        self.config = config or ActorConfig(name="enhanced_pool")
        
        # Metrics tracking
        self.metrics = PoolMetrics()
        
        # Create enhanced actors
        self.actors = []
        self._create_actors()
        
        # Create Ray's ActorPool with our actors
        self.ray_pool = RayActorPool(self.actors)
        
        # Optional work queue for buffering
        self.work_queue = None
        if enable_queue:
            self.work_queue = RayQueue(maxsize=queue_size)
            
        # Scalar operations for pool-level math
        self.scalar_ops = ScalarOperations(self._get_device())
        
        # Track initialization state
        self._initialized = False
        
    def _get_device(self) -> torch.device:
        """Get compute device for pool operations"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def _create_actors(self) -> None:
        """Create enhanced Ray actors"""
        for i in range(self.num_actors):
            # Create actor-specific config
            actor_config = ActorConfig(
                name=f"{self.actor_class.__name__}_{i}",
                num_gpus=self.config.num_gpus,
                num_cpus=self.config.num_cpus,
                max_concurrency=self.config.max_concurrency,
                memory_mb=self.config.memory_mb
            )
            
            # Create Ray remote actor
            if issubclass(self.actor_class, BaseActor):
                RemoteActor = self.actor_class.as_remote(actor_config)
            else:
                # Fallback for non-BaseActor classes
                RemoteActor = ray.remote(
                    num_gpus=actor_config.num_gpus,
                    num_cpus=actor_config.num_cpus,
                    max_concurrency=actor_config.max_concurrency
                )(self.actor_class)
            
            actor = RemoteActor.remote(f"actor_{i}", actor_config)
            self.actors.append(actor)
    
    async def initialize(self) -> None:
        """Initialize all actors in the pool"""
        if self._initialized:
            return
            
        # Initialize each actor
        init_tasks = []
        for actor in self.actors:
            if hasattr(actor, 'initialize'):
                init_tasks.append(actor.initialize.remote())
        
        if init_tasks:
            ray.get(init_tasks)
        
        self._initialized = True
    
    def map(self, fn: Callable, values: List[Any]) -> List[Any]:
        """
        Map function over values using Ray's ActorPool.
        Ensures all operations use PyTorch scalars.
        """
        start = time.perf_counter()
        
        # Convert numeric values to tensors
        tensor_values = []
        for v in values:
            if isinstance(v, (int, float)):
                tensor_values.append(self.scalar_ops.scalar(v))
            else:
                tensor_values.append(v)
        
        # Use Ray's ActorPool map
        results = list(self.ray_pool.map(fn, tensor_values))
        
        # Update metrics
        self.metrics.total_tasks_submitted += len(values)
        self.metrics.total_tasks_completed += len(results)
        elapsed = time.perf_counter() - start
        self.metrics.average_latency_ms = (elapsed * 1000) / len(values)
        
        return results
    
    def map_unordered(self, fn: Callable, values: List[Any]) -> List[Any]:
        """
        Map function over values without preserving order.
        More efficient for heterogeneous workloads.
        """
        # Convert to tensors
        tensor_values = []
        for v in values:
            if isinstance(v, (int, float)):
                tensor_values.append(self.scalar_ops.scalar(v))
            else:
                tensor_values.append(v)
        
        # Use Ray's unordered map for better performance
        results = list(self.ray_pool.map_unordered(fn, tensor_values))
        
        self.metrics.total_tasks_completed += len(results)
        
        return results
    
    def submit(self, fn: Callable, value: Any) -> None:
        """Submit single task to pool"""
        # Ensure PyTorch scalar
        if isinstance(value, (int, float)):
            value = self.scalar_ops.scalar(value)
            
        self.ray_pool.submit(fn, value)
        self.metrics.total_tasks_submitted += 1
    
    def get_next(self, timeout: Optional[float] = None) -> Any:
        """Get next result in order"""
        result = self.ray_pool.get_next(timeout=timeout)
        self.metrics.total_tasks_completed += 1
        return result
    
    def get_next_unordered(self, timeout: Optional[float] = None) -> Any:
        """Get any available result"""
        result = self.ray_pool.get_next_unordered(timeout=timeout)
        self.metrics.total_tasks_completed += 1
        return result
    
    def has_next(self) -> bool:
        """Check if results are pending"""
        return self.ray_pool.has_next()
    
    def has_free(self) -> bool:
        """Check if any actors are idle"""
        return self.ray_pool.has_free()
    
    async def broadcast_tensor(self, key: str, tensor: torch.Tensor) -> None:
        """Share tensor across all actors"""
        tasks = []
        for actor in self.actors:
            if hasattr(actor, 'share_tensor'):
                tasks.append(actor.share_tensor.remote(key, tensor))
        
        if tasks:
            ray.get(tasks)
            self.metrics.total_tensor_bytes += tensor.element_size() * tensor.nelement()
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect metrics from all actors"""
        health_tasks = []
        for actor in self.actors:
            if hasattr(actor, 'health_check'):
                health_tasks.append(actor.health_check.remote())
        
        if health_tasks:
            health_results = ray.get(health_tasks)
        else:
            health_results = []
        
        return {
            'pool_metrics': {
                'total_submitted': self.metrics.total_tasks_submitted,
                'total_completed': self.metrics.total_tasks_completed,
                'average_latency_ms': self.metrics.average_latency_ms,
                'tensor_bytes_shared': self.metrics.total_tensor_bytes
            },
            'actor_health': health_results,
            'num_actors': self.num_actors,
            'has_free': self.has_free(),
            'has_pending': self.has_next()
        }
    
    async def shutdown(self) -> None:
        """Graceful shutdown"""
        # Clear any pending work
        while self.has_next():
            try:
                self.get_next(timeout=0.1)
            except:
                break
        
        # Shutdown actors
        for actor in self.actors:
            if hasattr(actor, 'shutdown'):
                ray.get(actor.shutdown.remote())
            else:
                ray.kill(actor)
        
        self._initialized = False


class TensorStore:
    """
    Distributed tensor store for efficient sharing between actors.
    Uses Ray's object store for zero-copy tensor sharing.
    """
    
    def __init__(self):
        self.store = {}
        self.scalar_ops = ScalarOperations(self._get_device())
        
    def _get_device(self) -> torch.device:
        """Get compute device"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def put(self, key: str, tensor: torch.Tensor) -> ray.ObjectRef:
        """Store tensor in Ray's object store"""
        # Move tensor to CPU for Ray object store (required for serialization)
        if tensor.device.type != 'cpu':
            tensor = tensor.cpu()
            
        ref = ray.put(tensor)
        self.store[key] = ref
        return ref
    
    def get(self, key: str) -> Optional[torch.Tensor]:
        """Retrieve tensor from store"""
        if key in self.store:
            return ray.get(self.store[key])
        return None
    
    def delete(self, key: str) -> bool:
        """Remove tensor from store"""
        if key in self.store:
            del self.store[key]
            return True
        return False
    
    def list_keys(self) -> List[str]:
        """List all stored tensor keys"""
        return list(self.store.keys())
    
    def clear(self) -> None:
        """Clear all stored tensors"""
        self.store.clear()