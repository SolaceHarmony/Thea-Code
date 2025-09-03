"""
Wrappers for Actor-First Architecture
Core wrappers for Ray and PyTorch to ensure actor-first design
"""

from .ray_wrapper import (
    Actor,
    ActorHandle,
    ActorPool,
    ActorRegistry,
    remote_method,
    async_actor,
    get_actor,
    list_actors,
    kill_actor
)

from .torch_wrapper import (
    scalar,
    add,
    sub,
    mul,
    div,
    pow,
    sqrt,
    exp,
    log,
    sin,
    cos,
    mean,
    sum,
    min,
    max,
    TorchMath
)

__all__ = [
    # Ray wrappers
    'Actor',
    'ActorHandle', 
    'ActorPool',
    'ActorRegistry',
    'remote_method',
    'async_actor',
    'get_actor',
    'list_actors',
    'kill_actor',
    # PyTorch wrappers
    'scalar',
    'add',
    'sub',
    'mul',
    'div',
    'pow',
    'sqrt',
    'exp',
    'log',
    'sin',
    'cos',
    'mean',
    'sum',
    'min',
    'max',
    'TorchMath'
]