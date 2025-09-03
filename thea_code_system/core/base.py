#!/usr/bin/env python
"""
Base Classes for Production Architecture
Defines the fundamental class hierarchy
"""

import abc
import asyncio
import torch
import ray
from typing import Any, Dict, List, Optional, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime
import logging

# Type definitions
T = TypeVar('T')
ActorType = TypeVar('ActorType', bound='BaseActor')


@dataclass
class ActorConfig:
    """Configuration for actors"""
    name: str
    num_gpus: float = 0.25
    num_cpus: float = 1.0
    max_concurrency: int = 100
    memory_mb: Optional[int] = None
    enable_metrics: bool = True
    enable_health_checks: bool = True


@dataclass
class WorkerMetrics:
    """Metrics tracked by workers"""
    items_processed: int = 0
    errors_encountered: int = 0
    total_processing_time: float = 0.0
    last_activity: Optional[datetime] = None
    memory_usage_mb: float = 0.0
    
    def update(self, items: int = 1, time_taken: float = 0.0):
        """Update metrics"""
        self.items_processed += items
        self.total_processing_time += time_taken
        self.last_activity = datetime.now()
    
    @property
    def average_processing_time(self) -> float:
        """Calculate average processing time"""
        if self.items_processed == 0:
            return 0.0
        return self.total_processing_time / self.items_processed


class BaseWorker(abc.ABC):
    """
    Base class for all workers
    Workers are the fundamental processing units
    """
    
    def __init__(self, worker_id: str, config: Optional[ActorConfig] = None):
        self.worker_id = worker_id
        self.config = config or ActorConfig(name=worker_id)
        self.metrics = WorkerMetrics()
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{worker_id}")
        self.device = self._setup_device()
        self._initialized = False
        
    def _setup_device(self) -> torch.device:
        """Setup compute device"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    @abc.abstractmethod
    async def initialize(self) -> None:
        """Initialize worker resources"""
        pass
    
    @abc.abstractmethod
    async def process(self, data: Any) -> Any:
        """Process single item"""
        pass
    
    async def process_batch(self, batch: List[Any]) -> List[Any]:
        """Process batch of items"""
        results = []
        for item in batch:
            result = await self.process(item)
            results.append(result)
        return results
    
    async def health_check(self) -> Dict[str, Any]:
        """Check worker health"""
        return {
            'worker_id': self.worker_id,
            'healthy': self._initialized,
            'device': str(self.device),
            'metrics': {
                'items_processed': self.metrics.items_processed,
                'errors': self.metrics.errors_encountered,
                'avg_time': self.metrics.average_processing_time
            }
        }
    
    async def shutdown(self) -> None:
        """Graceful shutdown"""
        self._initialized = False
        self.logger.info(f"Worker {self.worker_id} shutting down")


class BaseActor(BaseWorker):
    """
    Base class for Ray actors
    Actors are distributed workers with state
    """
    
    def __init__(self, actor_id: str, config: Optional[ActorConfig] = None):
        super().__init__(worker_id=actor_id, config=config)
        self.actor_id = actor_id
        
    @classmethod
    def as_remote(cls, config: ActorConfig) -> ray.actor.ActorClass:
        """Create Ray remote actor class"""
        return ray.remote(
            num_gpus=config.num_gpus,
            num_cpus=config.num_cpus,
            max_concurrency=config.max_concurrency,
            memory=config.memory_mb * 1024 * 1024 if config.memory_mb else None
        )(cls)
    
    async def get_state(self) -> Dict[str, Any]:
        """Get actor state for checkpointing"""
        return {
            'actor_id': self.actor_id,
            'config': self.config,
            'metrics': self.metrics
        }
    
    async def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore actor state from checkpoint"""
        self.metrics = state.get('metrics', WorkerMetrics())


class BasePool(Generic[T]):
    """
    Base class for worker/actor pools
    Manages load balancing and distribution
    """
    
    def __init__(self, worker_class: type, size: int, config: Optional[ActorConfig] = None):
        self.worker_class = worker_class
        self.size = size
        self.config = config
        self.workers: List[T] = []
        self._current_index = 0
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialize pool workers"""
        for i in range(self.size):
            # Create config for each worker
            if self.config:
                # Copy config but override name
                config_dict = self.config.__dict__.copy()
                config_dict['name'] = f"{self.worker_class.__name__}_{i}"
                worker_config = ActorConfig(**config_dict)
            else:
                worker_config = ActorConfig(
                    name=f"{self.worker_class.__name__}_{i}"
                )
            
            if issubclass(self.worker_class, BaseActor):
                # Create Ray actor
                remote_class = self.worker_class.as_remote(worker_config)
                worker = remote_class.remote(f"worker_{i}", worker_config)
                # Initialize Ray actor
                await worker.initialize.remote()
            else:
                # Create regular worker
                worker = self.worker_class(f"worker_{i}", worker_config)
                await worker.initialize()
            
            self.workers.append(worker)
        
        self._initialized = True
    
    def get_next_worker(self) -> T:
        """Get next worker (round-robin)"""
        if not self._initialized:
            raise RuntimeError("Pool not initialized")
        
        worker = self.workers[self._current_index]
        self._current_index = (self._current_index + 1) % self.size
        return worker
    
    async def map(self, func_name: str, items: List[Any]) -> List[Any]:
        """Map function across items using pool"""
        tasks = []
        for item in items:
            worker = self.get_next_worker()
            
            if hasattr(worker, func_name):
                func = getattr(worker, func_name)
                if asyncio.iscoroutinefunction(func):
                    task = func(item)
                else:
                    # Handle Ray actors
                    task = func.remote(item) if hasattr(func, 'remote') else func(item)
                tasks.append(task)
        
        # Gather results
        if tasks and hasattr(tasks[0], '__ray_wait__'):
            # Ray futures
            results = ray.get(tasks)
        else:
            # Regular async
            results = await asyncio.gather(*tasks)
        
        return results
    
    async def broadcast(self, func_name: str, *args, **kwargs) -> List[Any]:
        """Broadcast function to all workers"""
        tasks = []
        for worker in self.workers:
            if hasattr(worker, func_name):
                func = getattr(worker, func_name)
                if asyncio.iscoroutinefunction(func):
                    task = func(*args, **kwargs)
                else:
                    task = func.remote(*args, **kwargs) if hasattr(func, 'remote') else func(*args, **kwargs)
                tasks.append(task)
        
        # Gather results
        if tasks and hasattr(tasks[0], '__ray_wait__'):
            results = ray.get(tasks)
        else:
            results = await asyncio.gather(*tasks)
        
        return results
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all workers"""
        health_results = await self.broadcast('health_check')
        
        healthy_count = sum(1 for r in health_results if r.get('healthy', False))
        
        return {
            'pool_size': self.size,
            'healthy_workers': healthy_count,
            'workers': health_results
        }
    
    async def shutdown(self) -> None:
        """Shutdown all workers"""
        await self.broadcast('shutdown')
        self._initialized = False


class BaseOrchestrator(abc.ABC):
    """
    Base class for orchestrators
    Orchestrators coordinate multiple pools and complex workflows
    """
    
    def __init__(self, name: str):
        self.name = name
        self.pools: Dict[str, BasePool] = {}
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{name}")
        self._initialized = False
    
    def register_pool(self, name: str, pool: BasePool) -> None:
        """Register a worker pool"""
        self.pools[name] = pool
    
    async def initialize(self) -> None:
        """Initialize all pools"""
        for name, pool in self.pools.items():
            self.logger.info(f"Initializing pool: {name}")
            await pool.initialize()
        self._initialized = True
    
    @abc.abstractmethod
    async def execute_workflow(self, input_data: Any) -> Any:
        """Execute orchestrated workflow"""
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of entire system"""
        pool_health = {}
        for name, pool in self.pools.items():
            pool_health[name] = await pool.health_check()
        
        return {
            'orchestrator': self.name,
            'initialized': self._initialized,
            'pools': pool_health
        }
    
    async def shutdown(self) -> None:
        """Shutdown all pools"""
        for name, pool in self.pools.items():
            self.logger.info(f"Shutting down pool: {name}")
            await pool.shutdown()
        self._initialized = False