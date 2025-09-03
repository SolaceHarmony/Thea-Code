#!/usr/bin/env python
"""
PyTorch Wrapper
Ensures ALL math operations use PyTorch scalars, no matter how trivial

Philosophy: No naked Python math. Ever. Everything is a tensor operation.
"""

import torch
from typing import Union, Optional, List, Tuple, Any
import math as python_math  # Keep for constants only

# Type for numeric inputs that will be converted to torch scalars
Numeric = Union[int, float, torch.Tensor]


class TorchMath:
    """
    Static class ensuring all math uses PyTorch
    Even 1+1 goes through PyTorch for GPU acceleration and consistency
    """
    
    # Default device - can be configured
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @classmethod
    def set_device(cls, device: Union[str, torch.device]) -> None:
        """Set default device for all operations"""
        cls.device = torch.device(device) if isinstance(device, str) else device
    
    @classmethod
    def scalar(cls, value: Numeric) -> torch.Tensor:
        """
        Convert any numeric value to PyTorch scalar
        This is THE fundamental operation - everything becomes a tensor
        """
        if torch.is_tensor(value):
            return value.to(cls.device)
        return torch.tensor(value, dtype=torch.float32, device=cls.device)
    
    @classmethod
    def add(cls, a: Numeric, b: Numeric) -> torch.Tensor:
        """Addition using PyTorch - even for 1+1"""
        a = cls.scalar(a)
        b = cls.scalar(b)
        return torch.add(a, b)
    
    @classmethod
    def sub(cls, a: Numeric, b: Numeric) -> torch.Tensor:
        """Subtraction using PyTorch"""
        a = cls.scalar(a)
        b = cls.scalar(b)
        return torch.sub(a, b)
    
    @classmethod
    def mul(cls, a: Numeric, b: Numeric) -> torch.Tensor:
        """Multiplication using PyTorch"""
        a = cls.scalar(a)
        b = cls.scalar(b)
        return torch.mul(a, b)
    
    @classmethod
    def div(cls, a: Numeric, b: Numeric) -> torch.Tensor:
        """Division using PyTorch"""
        a = cls.scalar(a)
        b = cls.scalar(b)
        return torch.div(a, b)
    
    @classmethod
    def pow(cls, a: Numeric, b: Numeric) -> torch.Tensor:
        """Power operation using PyTorch"""
        a = cls.scalar(a)
        b = cls.scalar(b)
        return torch.pow(a, b)
    
    @classmethod
    def sqrt(cls, a: Numeric) -> torch.Tensor:
        """Square root using PyTorch"""
        a = cls.scalar(a)
        return torch.sqrt(a)
    
    @classmethod
    def exp(cls, a: Numeric) -> torch.Tensor:
        """Exponential using PyTorch"""
        a = cls.scalar(a)
        return torch.exp(a)
    
    @classmethod
    def log(cls, a: Numeric) -> torch.Tensor:
        """Natural logarithm using PyTorch"""
        a = cls.scalar(a)
        return torch.log(a)
    
    @classmethod
    def sin(cls, a: Numeric) -> torch.Tensor:
        """Sine using PyTorch"""
        a = cls.scalar(a)
        return torch.sin(a)
    
    @classmethod
    def cos(cls, a: Numeric) -> torch.Tensor:
        """Cosine using PyTorch"""
        a = cls.scalar(a)
        return torch.cos(a)
    
    @classmethod
    def tan(cls, a: Numeric) -> torch.Tensor:
        """Tangent using PyTorch"""
        a = cls.scalar(a)
        return torch.tan(a)
    
    @classmethod
    def abs(cls, a: Numeric) -> torch.Tensor:
        """Absolute value using PyTorch"""
        a = cls.scalar(a)
        return torch.abs(a)
    
    @classmethod
    def min(cls, *args: Numeric) -> torch.Tensor:
        """Minimum using PyTorch"""
        tensors = [cls.scalar(a) for a in args]
        if len(tensors) == 1:
            return tensors[0]
        stacked = torch.stack(tensors)
        return torch.min(stacked)
    
    @classmethod
    def max(cls, *args: Numeric) -> torch.Tensor:
        """Maximum using PyTorch"""
        tensors = [cls.scalar(a) for a in args]
        if len(tensors) == 1:
            return tensors[0]
        stacked = torch.stack(tensors)
        return torch.max(stacked)
    
    @classmethod
    def mean(cls, *args: Numeric) -> torch.Tensor:
        """Mean using PyTorch"""
        tensors = [cls.scalar(a) for a in args]
        stacked = torch.stack(tensors)
        return torch.mean(stacked)
    
    @classmethod
    def sum(cls, *args: Numeric) -> torch.Tensor:
        """Sum using PyTorch"""
        tensors = [cls.scalar(a) for a in args]
        stacked = torch.stack(tensors)
        return torch.sum(stacked)
    
    @classmethod
    def floor(cls, a: Numeric) -> torch.Tensor:
        """Floor using PyTorch"""
        a = cls.scalar(a)
        return torch.floor(a)
    
    @classmethod
    def ceil(cls, a: Numeric) -> torch.Tensor:
        """Ceiling using PyTorch"""
        a = cls.scalar(a)
        return torch.ceil(a)
    
    @classmethod
    def round(cls, a: Numeric) -> torch.Tensor:
        """Round using PyTorch"""
        a = cls.scalar(a)
        return torch.round(a)
    
    @classmethod
    def mod(cls, a: Numeric, b: Numeric) -> torch.Tensor:
        """Modulo using PyTorch"""
        a = cls.scalar(a)
        b = cls.scalar(b)
        return torch.fmod(a, b)
    
    @classmethod
    def eq(cls, a: Numeric, b: Numeric) -> torch.Tensor:
        """Equality comparison using PyTorch"""
        a = cls.scalar(a)
        b = cls.scalar(b)
        return torch.eq(a, b)
    
    @classmethod
    def ne(cls, a: Numeric, b: Numeric) -> torch.Tensor:
        """Not equal comparison using PyTorch"""
        a = cls.scalar(a)
        b = cls.scalar(b)
        return torch.ne(a, b)
    
    @classmethod
    def lt(cls, a: Numeric, b: Numeric) -> torch.Tensor:
        """Less than comparison using PyTorch"""
        a = cls.scalar(a)
        b = cls.scalar(b)
        return torch.lt(a, b)
    
    @classmethod
    def le(cls, a: Numeric, b: Numeric) -> torch.Tensor:
        """Less than or equal comparison using PyTorch"""
        a = cls.scalar(a)
        b = cls.scalar(b)
        return torch.le(a, b)
    
    @classmethod
    def gt(cls, a: Numeric, b: Numeric) -> torch.Tensor:
        """Greater than comparison using PyTorch"""
        a = cls.scalar(a)
        b = cls.scalar(b)
        return torch.gt(a, b)
    
    @classmethod
    def ge(cls, a: Numeric, b: Numeric) -> torch.Tensor:
        """Greater than or equal comparison using PyTorch"""
        a = cls.scalar(a)
        b = cls.scalar(b)
        return torch.ge(a, b)
    
    @classmethod
    def clamp(cls, a: Numeric, min_val: Numeric, max_val: Numeric) -> torch.Tensor:
        """Clamp value between min and max using PyTorch"""
        a = cls.scalar(a)
        min_val = cls.scalar(min_val)
        max_val = cls.scalar(max_val)
        return torch.clamp(a, min=min_val, max=max_val)
    
    @classmethod
    def sigmoid(cls, a: Numeric) -> torch.Tensor:
        """Sigmoid activation using PyTorch"""
        a = cls.scalar(a)
        return torch.sigmoid(a)
    
    @classmethod
    def tanh(cls, a: Numeric) -> torch.Tensor:
        """Tanh activation using PyTorch"""
        a = cls.scalar(a)
        return torch.tanh(a)
    
    @classmethod
    def relu(cls, a: Numeric) -> torch.Tensor:
        """ReLU activation using PyTorch"""
        a = cls.scalar(a)
        return torch.relu(a)
    
    # Constants as PyTorch scalars
    @classmethod
    def pi(cls) -> torch.Tensor:
        """Pi as PyTorch scalar"""
        return cls.scalar(python_math.pi)
    
    @classmethod
    def e(cls) -> torch.Tensor:
        """Euler's number as PyTorch scalar"""
        return cls.scalar(python_math.e)
    
    @classmethod
    def inf(cls) -> torch.Tensor:
        """Infinity as PyTorch scalar"""
        return cls.scalar(float('inf'))
    
    @classmethod
    def nan(cls) -> torch.Tensor:
        """NaN as PyTorch scalar"""
        return cls.scalar(float('nan'))


# Convenience functions that mirror TorchMath methods
# These allow direct import and use: from torch_wrapper import add, mul, etc.

def scalar(value: Numeric) -> torch.Tensor:
    """Convert to PyTorch scalar"""
    return TorchMath.scalar(value)

def add(a: Numeric, b: Numeric) -> torch.Tensor:
    """Add using PyTorch"""
    return TorchMath.add(a, b)

def sub(a: Numeric, b: Numeric) -> torch.Tensor:
    """Subtract using PyTorch"""
    return TorchMath.sub(a, b)

def mul(a: Numeric, b: Numeric) -> torch.Tensor:
    """Multiply using PyTorch"""
    return TorchMath.mul(a, b)

def div(a: Numeric, b: Numeric) -> torch.Tensor:
    """Divide using PyTorch"""
    return TorchMath.div(a, b)

def pow(a: Numeric, b: Numeric) -> torch.Tensor:
    """Power using PyTorch"""
    return TorchMath.pow(a, b)

def sqrt(a: Numeric) -> torch.Tensor:
    """Square root using PyTorch"""
    return TorchMath.sqrt(a)

def exp(a: Numeric) -> torch.Tensor:
    """Exponential using PyTorch"""
    return TorchMath.exp(a)

def log(a: Numeric) -> torch.Tensor:
    """Natural log using PyTorch"""
    return TorchMath.log(a)

def sin(a: Numeric) -> torch.Tensor:
    """Sine using PyTorch"""
    return TorchMath.sin(a)

def cos(a: Numeric) -> torch.Tensor:
    """Cosine using PyTorch"""
    return TorchMath.cos(a)

def mean(*args: Numeric) -> torch.Tensor:
    """Mean using PyTorch"""
    return TorchMath.mean(*args)

def sum(*args: Numeric) -> torch.Tensor:
    """Sum using PyTorch"""
    return TorchMath.sum(*args)

def min(*args: Numeric) -> torch.Tensor:
    """Minimum using PyTorch"""
    return TorchMath.min(*args)

def max(*args: Numeric) -> torch.Tensor:
    """Maximum using PyTorch"""
    return TorchMath.max(*args)


class TorchCounter:
    """
    Counter that uses PyTorch scalars for all operations
    Even simple counting uses GPU acceleration
    """
    
    def __init__(self, initial: Numeric = 0, device: Optional[torch.device] = None):
        """Initialize counter with PyTorch scalar"""
        self.device = device or TorchMath.device
        self.value = torch.tensor(initial, dtype=torch.float32, device=self.device)
    
    def increment(self, amount: Numeric = 1) -> torch.Tensor:
        """Increment counter using PyTorch"""
        amount = TorchMath.scalar(amount)
        self.value = torch.add(self.value, amount)
        return self.value
    
    def decrement(self, amount: Numeric = 1) -> torch.Tensor:
        """Decrement counter using PyTorch"""
        amount = TorchMath.scalar(amount)
        self.value = torch.sub(self.value, amount)
        return self.value
    
    def reset(self) -> torch.Tensor:
        """Reset counter to zero using PyTorch"""
        self.value = torch.zeros(1, device=self.device)
        return self.value
    
    def get(self) -> torch.Tensor:
        """Get current value"""
        return self.value


class TorchAccumulator:
    """
    Accumulator that uses PyTorch for all operations
    Useful for metrics, running averages, etc.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """Initialize accumulator"""
        self.device = device or TorchMath.device
        self.values: List[torch.Tensor] = []
        self.sum = torch.zeros(1, device=self.device)
        self.count = torch.zeros(1, device=self.device)
    
    def add(self, value: Numeric) -> None:
        """Add value to accumulator"""
        value = TorchMath.scalar(value)
        self.values.append(value)
        self.sum = torch.add(self.sum, value)
        self.count = torch.add(self.count, TorchMath.scalar(1))
    
    def mean(self) -> torch.Tensor:
        """Get mean of accumulated values"""
        if torch.eq(self.count, TorchMath.scalar(0)):
            return torch.zeros(1, device=self.device)
        return torch.div(self.sum, self.count)
    
    def total(self) -> torch.Tensor:
        """Get sum of accumulated values"""
        return self.sum
    
    def size(self) -> torch.Tensor:
        """Get count of accumulated values"""
        return self.count
    
    def reset(self) -> None:
        """Reset accumulator"""
        self.values = []
        self.sum = torch.zeros(1, device=self.device)
        self.count = torch.zeros(1, device=self.device)


# Example usage patterns
def example_usage():
    """
    Demonstrates the PyTorch-everywhere philosophy
    Even trivial operations use PyTorch scalars
    """
    
    # Simple arithmetic - ALL using PyTorch
    one = scalar(1)
    two = scalar(2)
    three = add(one, two)  # Even 1+2 uses PyTorch!
    
    # Comparisons use PyTorch
    is_equal = TorchMath.eq(three, scalar(3))
    
    # Loops with PyTorch counters
    counter = TorchCounter()
    for _ in range(10):
        counter.increment()  # GPU-accelerated counting!
    
    # Accumulate metrics
    acc = TorchAccumulator()
    for i in range(100):
        acc.add(i)  # All additions on GPU
    
    mean_value = acc.mean()  # Mean computed on GPU
    
    return {
        'three': three,
        'is_equal': is_equal,
        'count': counter.get(),
        'mean': mean_value
    }


# Set default device on import
if torch.cuda.is_available():
    TorchMath.set_device("cuda")
elif torch.backends.mps.is_available():
    TorchMath.set_device("mps")
else:
    TorchMath.set_device("cpu")