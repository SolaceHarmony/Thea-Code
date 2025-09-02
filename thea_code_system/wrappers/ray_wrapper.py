#!/usr/bin/env python
"""
Ray Actor Wrapper
Actor-first wrapper around Ray for clean actor-centric design

All computation happens in actors. No exceptions.
"""

import ray
import asyncio
from typing import Any, Dict, List, Optional, Type, TypeVar, Callable
from functools import wraps
import torch

# Type for actor classes
ActorType = TypeVar('ActorType')


class ActorRegistry:
    """
    Central registry for all actors in the system
    Enforces actor-first principle: everything is an actor
    """
    
    _actors: Dict[str, ray.ActorHandle] = {}
    _actor_types: Dict[str, Type] = {}
    
    @classmethod
    def register(cls, name: str, actor_handle: ray.ActorHandle, actor_type: Type) -> None:
        """Register an actor in the system"""
        cls._actors[name] = actor_handle
        cls._actor_types[name] = actor_type
    
    @classmethod
    def get(cls, name: str) -> Optional[ray.ActorHandle]:
        """Get actor by name"""
        return cls._actors.get(name)
    
    @classmethod
    def list(cls) -> List[str]:
        """List all registered actors"""
        return list(cls._actors.keys())
    
    @classmethod
    def kill(cls, name: str) -> bool:
        """Kill an actor and remove from registry"""
        if name in cls._actors:
            ray.kill(cls._actors[name])
            del cls._actors[name]
            del cls._actor_types[name]
            return True
        return False
    
    @classmethod
    def clear(cls) -> None:
        """Kill all actors and clear registry"""
        for name in list(cls._actors.keys()):
            cls.kill(name)


class Actor:
    """
    Base class for all actors in the system
    Enforces actor-first design principles
    """
    
    def __init__(self, name: str = None):
        """Initialize actor with optional name"""
        self.name = name or self.__class__.__name__
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialized = True
    
    async def health_check(self) -> Dict[str, Any]:
        """Standard health check for all actors"""
        return {
            'actor': self.name,
            'healthy': True,
            'device': str(self.device),
            'initialized': self._initialized
        }
    
    async def shutdown(self) -> None:
        """Graceful shutdown"""
        self._initialized = False


def async_actor(num_gpus: float = 0, num_cpus: float = 1, max_concurrency: int = 100):
    """
    Decorator to create async Ray actors with proper resource allocation
    
    Example:
        @async_actor(num_gpus=0.25)
        class MyActor(Actor):
            async def process(self, data):
                return data
    """
    def decorator(cls: Type[ActorType]) -> Type[ActorType]:
        # Ensure it inherits from Actor
        if not issubclass(cls, Actor):
            raise TypeError(f"{cls.__name__} must inherit from Actor base class")
        
        # Create Ray remote actor with resources
        remote_cls = ray.remote(
            num_gpus=num_gpus,
            num_cpus=num_cpus,
            max_concurrency=max_concurrency
        )(cls)
        
        # Store original class reference
        remote_cls._original_class = cls
        
        return remote_cls
    
    return decorator


def remote_method(func: Callable) -> Callable:
    """
    Decorator for actor methods to ensure they're async and remote-ready
    Automatically handles PyTorch tensors and scalars
    """
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        # Ensure we're in actor context
        if not hasattr(self, '_initialized'):
            raise RuntimeError(f"Method {func.__name__} must be called on an Actor instance")
        
        # Execute the method
        result = await func(self, *args, **kwargs)
        
        # Handle tensor returns properly
        if torch.is_tensor(result):
            # Move to CPU for serialization if needed
            if result.is_cuda:
                result = result.cpu()
        
        return result
    
    return wrapper


class ActorHandle:
    """
    Wrapper around Ray ActorHandle for cleaner interface
    """
    
    def __init__(self, ray_handle: ray.ActorHandle, name: str, actor_type: Type):
        self._ray_handle = ray_handle
        self.name = name
        self.actor_type = actor_type
        
    def __getattr__(self, name: str) -> Callable:
        """Proxy method calls to Ray actor"""
        return getattr(self._ray_handle, name)
    
    async def call(self, method_name: str, *args, **kwargs) -> Any:
        """Call a method on the actor"""
        method = getattr(self._ray_handle, method_name)
        return await method.remote(*args, **kwargs)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check actor health"""
        return await self._ray_handle.health_check.remote()
    
    def kill(self) -> None:
        """Kill this actor"""
        ActorRegistry.kill(self.name)


class ActorPool:
    """
    Pool of actors for load balancing
    Ensures all work is distributed across actors
    """
    
    def __init__(self, actor_class: Type[ActorType], num_actors: int, **actor_kwargs):
        """Create a pool of actors"""
        self.actor_class = actor_class
        self.num_actors = num_actors
        self.actors: List[ActorHandle] = []
        self._current = 0
        
        # Create actors
        for i in range(num_actors):
            name = f"{actor_class.__name__}_pool_{i}"
            ray_handle = actor_class.remote(name=name, **actor_kwargs)
            handle = ActorHandle(ray_handle, name, actor_class)
            self.actors.append(handle)
            ActorRegistry.register(name, ray_handle, actor_class)
    
    def get_next(self) -> ActorHandle:
        """Get next actor in round-robin fashion"""
        actor = self.actors[self._current]
        self._current = (self._current + 1) % self.num_actors
        return actor
    
    async def map(self, method_name: str, items: List[Any]) -> List[Any]:
        """Map a method across all items using the actor pool"""
        tasks = []
        for item in items:
            actor = self.get_next()
            task = actor.call(method_name, item)
            tasks.append(task)
        
        return await asyncio.gather(*tasks)
    
    async def broadcast(self, method_name: str, *args, **kwargs) -> List[Any]:
        """Call a method on all actors in the pool"""
        tasks = [actor.call(method_name, *args, **kwargs) for actor in self.actors]
        return await asyncio.gather(*tasks)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all actors in pool"""
        results = await self.broadcast("health_check")
        return {
            'pool_size': self.num_actors,
            'healthy_actors': sum(1 for r in results if r['healthy']),
            'actors': results
        }
    
    def shutdown(self) -> None:
        """Shutdown all actors in pool"""
        for actor in self.actors:
            actor.kill()


# Convenience functions
def create_actor(actor_class: Type[ActorType], name: str = None, **kwargs) -> ActorHandle:
    """Create a single actor and register it"""
    name = name or actor_class.__name__
    ray_handle = actor_class.remote(name=name, **kwargs)
    handle = ActorHandle(ray_handle, name, actor_class)
    ActorRegistry.register(name, ray_handle, actor_class)
    return handle


def get_actor(name: str) -> Optional[ActorHandle]:
    """Get an actor by name"""
    ray_handle = ActorRegistry.get(name)
    if ray_handle:
        actor_type = ActorRegistry._actor_types[name]
        return ActorHandle(ray_handle, name, actor_type)
    return None


def list_actors() -> List[str]:
    """List all registered actors"""
    return ActorRegistry.list()


def kill_actor(name: str) -> bool:
    """Kill an actor by name"""
    return ActorRegistry.kill(name)


# Initialize Ray if not already initialized
def ensure_ray_initialized():
    """Ensure Ray is initialized for actor-first operation"""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)


# Auto-initialize on import for actor-first design
ensure_ray_initialized()