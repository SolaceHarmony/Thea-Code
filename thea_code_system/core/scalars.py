#!/usr/bin/env python
"""
PyTorch Scalar Operations
Ensures all mathematical operations use PyTorch tensors
"""

import torch
from typing import Union, List, Optional, Any, Dict
from dataclasses import dataclass
import logging


# Type for numeric inputs
Numeric = Union[int, float, torch.Tensor]


class ScalarOperations:
    """
    All mathematical operations use PyTorch scalars
    Even trivial operations like 1+1 go through PyTorch
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or self._get_default_device()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _get_default_device(self) -> torch.device:
        """Get best available device"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def scalar(self, value: Numeric) -> torch.Tensor:
        """Convert any numeric to PyTorch scalar"""
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        return torch.tensor(value, dtype=torch.float32, device=self.device)
    
    # Arithmetic operations
    def add(self, a: Numeric, b: Numeric) -> torch.Tensor:
        """Addition using PyTorch"""
        return torch.add(self.scalar(a), self.scalar(b))
    
    def sub(self, a: Numeric, b: Numeric) -> torch.Tensor:
        """Subtraction using PyTorch"""
        return torch.sub(self.scalar(a), self.scalar(b))
    
    def mul(self, a: Numeric, b: Numeric) -> torch.Tensor:
        """Multiplication using PyTorch"""
        return torch.mul(self.scalar(a), self.scalar(b))
    
    def div(self, a: Numeric, b: Numeric) -> torch.Tensor:
        """Division using PyTorch"""
        return torch.div(self.scalar(a), self.scalar(b))
    
    def pow(self, a: Numeric, b: Numeric) -> torch.Tensor:
        """Power using PyTorch"""
        return torch.pow(self.scalar(a), self.scalar(b))
    
    def mod(self, a: Numeric, b: Numeric) -> torch.Tensor:
        """Modulo using PyTorch"""
        return torch.fmod(self.scalar(a), self.scalar(b))
    
    # Mathematical functions
    def sqrt(self, a: Numeric) -> torch.Tensor:
        """Square root using PyTorch"""
        return torch.sqrt(self.scalar(a))
    
    def exp(self, a: Numeric) -> torch.Tensor:
        """Exponential using PyTorch"""
        return torch.exp(self.scalar(a))
    
    def log(self, a: Numeric) -> torch.Tensor:
        """Natural logarithm using PyTorch"""
        return torch.log(self.scalar(a))
    
    def abs(self, a: Numeric) -> torch.Tensor:
        """Absolute value using PyTorch"""
        return torch.abs(self.scalar(a))
    
    def floor(self, a: Numeric) -> torch.Tensor:
        """Floor using PyTorch"""
        return torch.floor(self.scalar(a))
    
    def ceil(self, a: Numeric) -> torch.Tensor:
        """Ceiling using PyTorch"""
        return torch.ceil(self.scalar(a))
    
    def round(self, a: Numeric) -> torch.Tensor:
        """Round using PyTorch"""
        return torch.round(self.scalar(a))
    
    # Trigonometric functions
    def sin(self, a: Numeric) -> torch.Tensor:
        """Sine using PyTorch"""
        return torch.sin(self.scalar(a))
    
    def cos(self, a: Numeric) -> torch.Tensor:
        """Cosine using PyTorch"""
        return torch.cos(self.scalar(a))
    
    def tan(self, a: Numeric) -> torch.Tensor:
        """Tangent using PyTorch"""
        return torch.tan(self.scalar(a))
    
    # Comparison operations
    def eq(self, a: Numeric, b: Numeric) -> torch.Tensor:
        """Equality using PyTorch"""
        return torch.eq(self.scalar(a), self.scalar(b))
    
    def ne(self, a: Numeric, b: Numeric) -> torch.Tensor:
        """Not equal using PyTorch"""
        return torch.ne(self.scalar(a), self.scalar(b))
    
    def lt(self, a: Numeric, b: Numeric) -> torch.Tensor:
        """Less than using PyTorch"""
        return torch.lt(self.scalar(a), self.scalar(b))
    
    def le(self, a: Numeric, b: Numeric) -> torch.Tensor:
        """Less than or equal using PyTorch"""
        return torch.le(self.scalar(a), self.scalar(b))
    
    def gt(self, a: Numeric, b: Numeric) -> torch.Tensor:
        """Greater than using PyTorch"""
        return torch.gt(self.scalar(a), self.scalar(b))
    
    def ge(self, a: Numeric, b: Numeric) -> torch.Tensor:
        """Greater than or equal using PyTorch"""
        return torch.ge(self.scalar(a), self.scalar(b))
    
    # Aggregation operations
    def min(self, *values: Numeric) -> torch.Tensor:
        """Minimum using PyTorch"""
        tensors = [self.scalar(v) for v in values]
        if len(tensors) == 1:
            return tensors[0]
        return torch.min(torch.stack(tensors))
    
    def max(self, *values: Numeric) -> torch.Tensor:
        """Maximum using PyTorch"""
        tensors = [self.scalar(v) for v in values]
        if len(tensors) == 1:
            return tensors[0]
        return torch.max(torch.stack(tensors))
    
    def mean(self, *values: Numeric) -> torch.Tensor:
        """Mean using PyTorch"""
        tensors = [self.scalar(v) for v in values]
        return torch.mean(torch.stack(tensors))
    
    def sum(self, *values: Numeric) -> torch.Tensor:
        """Sum using PyTorch"""
        tensors = [self.scalar(v) for v in values]
        return torch.sum(torch.stack(tensors))
    
    def std(self, *values: Numeric) -> torch.Tensor:
        """Standard deviation using PyTorch"""
        tensors = [self.scalar(v) for v in values]
        return torch.std(torch.stack(tensors))
    
    # Utility operations
    def clamp(self, value: Numeric, min_val: Numeric, max_val: Numeric) -> torch.Tensor:
        """Clamp value between min and max"""
        return torch.clamp(
            self.scalar(value),
            min=self.scalar(min_val),
            max=self.scalar(max_val)
        )
    
    def sigmoid(self, a: Numeric) -> torch.Tensor:
        """Sigmoid activation"""
        return torch.sigmoid(self.scalar(a))
    
    def relu(self, a: Numeric) -> torch.Tensor:
        """ReLU activation"""
        return torch.relu(self.scalar(a))
    
    def tanh(self, a: Numeric) -> torch.Tensor:
        """Tanh activation"""
        return torch.tanh(self.scalar(a))


class TensorMetrics:
    """
    Metrics tracking using PyTorch tensors
    All counters and accumulators use GPU
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or self._get_default_device()
        self.ops = ScalarOperations(self.device)
        self._metrics = {}
    
    def _get_default_device(self) -> torch.device:
        """Get best available device"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def create_counter(self, name: str, initial: Numeric = 0) -> None:
        """Create a new counter"""
        self._metrics[name] = self.ops.scalar(initial)
    
    def increment(self, name: str, amount: Numeric = 1) -> torch.Tensor:
        """Increment counter"""
        if name not in self._metrics:
            self.create_counter(name)
        
        self._metrics[name] = self.ops.add(self._metrics[name], amount)
        return self._metrics[name]
    
    def decrement(self, name: str, amount: Numeric = 1) -> torch.Tensor:
        """Decrement counter"""
        if name not in self._metrics:
            self.create_counter(name)
        
        self._metrics[name] = self.ops.sub(self._metrics[name], amount)
        return self._metrics[name]
    
    def get(self, name: str) -> Optional[torch.Tensor]:
        """Get metric value"""
        return self._metrics.get(name)
    
    def get_all(self) -> Dict[str, float]:
        """Get all metrics as Python floats"""
        return {
            name: value.item() if isinstance(value, torch.Tensor) else value
            for name, value in self._metrics.items()
        }
    
    def reset(self, name: Optional[str] = None) -> None:
        """Reset metric(s)"""
        if name:
            if name in self._metrics:
                self._metrics[name] = self.ops.scalar(0)
        else:
            for name in self._metrics:
                self._metrics[name] = self.ops.scalar(0)


class AccumulatorBase:
    """
    Base class for accumulators using PyTorch tensors
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or self._get_default_device()
        self.ops = ScalarOperations(self.device)
        self.values: List[torch.Tensor] = []
        self._sum = self.ops.scalar(0)
        self._count = self.ops.scalar(0)
    
    def _get_default_device(self) -> torch.device:
        """Get best available device"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def add(self, value: Numeric) -> None:
        """Add value to accumulator"""
        tensor_value = self.ops.scalar(value)
        self.values.append(tensor_value)
        self._sum = self.ops.add(self._sum, tensor_value)
        self._count = self.ops.add(self._count, 1)
    
    def add_batch(self, values: List[Numeric]) -> None:
        """Add batch of values"""
        for value in values:
            self.add(value)
    
    def mean(self) -> torch.Tensor:
        """Calculate mean"""
        if self.ops.eq(self._count, 0):
            return self.ops.scalar(0)
        return self.ops.div(self._sum, self._count)
    
    def sum(self) -> torch.Tensor:
        """Get sum"""
        return self._sum
    
    def count(self) -> torch.Tensor:
        """Get count"""
        return self._count
    
    def min(self) -> torch.Tensor:
        """Get minimum value"""
        if not self.values:
            return self.ops.scalar(float('inf'))
        return torch.min(torch.stack(self.values))
    
    def max(self) -> torch.Tensor:
        """Get maximum value"""
        if not self.values:
            return self.ops.scalar(float('-inf'))
        return torch.max(torch.stack(self.values))
    
    def std(self) -> torch.Tensor:
        """Calculate standard deviation"""
        if len(self.values) < 2:
            return self.ops.scalar(0)
        return torch.std(torch.stack(self.values))
    
    def reset(self) -> None:
        """Reset accumulator"""
        self.values = []
        self._sum = self.ops.scalar(0)
        self._count = self.ops.scalar(0)
    
    def get_stats(self) -> Dict[str, float]:
        """Get all statistics"""
        if not self.values:
            return {
                'count': 0,
                'sum': 0,
                'mean': 0,
                'min': 0,
                'max': 0,
                'std': 0
            }
        
        return {
            'count': self.count().item(),
            'sum': self.sum().item(),
            'mean': self.mean().item(),
            'min': self.min().item(),
            'max': self.max().item(),
            'std': self.std().item()
        }