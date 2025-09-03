#!/usr/bin/env python
"""
MPS-Specific Components
Leverages Apple Silicon Metal Performance Shaders
"""

import torch
import torch.mps
import torch.nn as nn
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import logging
from contextlib import contextmanager


@dataclass
class MPSConfig:
    """MPS-specific configuration"""
    memory_fraction: float = 0.8
    enable_profiling: bool = False
    manual_seed: Optional[int] = None
    synchronize_after_operations: bool = False
    empty_cache_interval: int = 100  # Operations between cache clears


class MPSDevice:
    """
    Manages MPS device operations
    Singleton pattern for device management
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.logger = logging.getLogger(self.__class__.__name__)
            self.device = None
            self.config = MPSConfig()
            self._operation_count = 0
            self._initialize()
            self._initialized = True
    
    def _initialize(self):
        """Initialize MPS device"""
        if not torch.backends.mps.is_available():
            self.logger.warning("MPS not available, using CPU")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("mps")
            self.logger.info(f"MPS initialized: {torch.mps.device_count()} device(s)")
            
            # Apply configuration
            torch.mps.set_per_process_memory_fraction(self.config.memory_fraction)
            
            if self.config.manual_seed is not None:
                torch.mps.manual_seed(self.config.manual_seed)
    
    def configure(self, config: MPSConfig):
        """Update MPS configuration"""
        self.config = config
        
        if self.device.type == "mps":
            torch.mps.set_per_process_memory_fraction(config.memory_fraction)
            if config.manual_seed is not None:
                torch.mps.manual_seed(config.manual_seed)
    
    @property
    def is_available(self) -> bool:
        """Check if MPS is available"""
        return self.device.type == "mps"
    
    def synchronize(self):
        """Synchronize MPS operations"""
        if self.device.type == "mps":
            torch.mps.synchronize()
    
    def empty_cache(self):
        """Empty MPS cache"""
        if self.device.type == "mps":
            torch.mps.empty_cache()
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics"""
        if self.device.type != "mps":
            return {}
        
        return {
            'current_allocated_mb': torch.mps.current_allocated_memory() / 1024**2,
            'driver_allocated_mb': torch.mps.driver_allocated_memory() / 1024**2,
            'recommended_max_gb': torch.mps.recommended_max_memory() / 1024**3
        }
    
    @contextmanager
    def operation_context(self):
        """Context manager for MPS operations"""
        try:
            yield self
            
            self._operation_count += 1
            
            # Periodic cache clearing
            if self._operation_count % self.config.empty_cache_interval == 0:
                self.empty_cache()
            
            # Optional synchronization
            if self.config.synchronize_after_operations:
                self.synchronize()
                
        except Exception as e:
            self.logger.error(f"MPS operation failed: {e}")
            raise


class MPSMemoryManager:
    """
    Manages memory allocation and optimization for MPS
    """
    
    def __init__(self, device: MPSDevice):
        self.device = device
        self.logger = logging.getLogger(self.__class__.__name__)
        self._allocation_history = []
        self._peak_memory = 0
    
    def allocate_tensor(self, *shape: int, dtype=torch.float32) -> torch.Tensor:
        """Allocate tensor with memory tracking"""
        with self.device.operation_context():
            tensor = torch.empty(*shape, dtype=dtype, device=self.device.device)
            
            if self.device.is_available:
                current_mem = torch.mps.current_allocated_memory()
                self._allocation_history.append(current_mem)
                self._peak_memory = max(self._peak_memory, current_mem)
            
            return tensor
    
    def optimize_memory(self):
        """Optimize memory usage"""
        if not self.device.is_available:
            return
        
        initial_mem = torch.mps.current_allocated_memory()
        
        # Clear cache
        torch.mps.empty_cache()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        final_mem = torch.mps.current_allocated_memory()
        freed = initial_mem - final_mem
        
        self.logger.info(f"Memory optimized: {freed / 1024**2:.2f} MB freed")
    
    def get_allocation_report(self) -> Dict[str, Any]:
        """Get memory allocation report"""
        if not self._allocation_history:
            return {}
        
        return {
            'allocations': len(self._allocation_history),
            'peak_memory_mb': self._peak_memory / 1024**2,
            'current_memory_mb': torch.mps.current_allocated_memory() / 1024**2 if self.device.is_available else 0,
            'average_allocation_mb': sum(self._allocation_history) / len(self._allocation_history) / 1024**2
        }


class MPSStreamManager:
    """
    Manages parallel execution streams on MPS
    Note: MPS uses implicit streaming, this class provides abstraction
    """
    
    def __init__(self, device: MPSDevice):
        self.device = device
        self.logger = logging.getLogger(self.__class__.__name__)
        self._operations_queued = []
    
    def queue_operation(self, operation: Callable, *args, **kwargs) -> Any:
        """
        Queue operation for parallel execution
        Operations are executed asynchronously on MPS
        """
        with self.device.operation_context():
            # Execute operation (queued on MPS, not blocking)
            result = operation(*args, **kwargs)
            self._operations_queued.append(result)
            return result
    
    def queue_parallel_operations(self, operations: List[tuple]) -> List[Any]:
        """
        Queue multiple operations for parallel execution
        Each tuple is (operation, args, kwargs)
        """
        results = []
        
        with self.device.operation_context():
            for op, args, kwargs in operations:
                result = op(*args, **kwargs)
                results.append(result)
                self._operations_queued.append(result)
        
        return results
    
    def synchronize_all(self) -> None:
        """Wait for all queued operations to complete"""
        self.device.synchronize()
        self._operations_queued.clear()
    
    def parallel_matmul(self, matrices_a: List[torch.Tensor], matrices_b: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Perform parallel matrix multiplications
        Leverages MPS implicit streaming
        """
        if len(matrices_a) != len(matrices_b):
            raise ValueError("Matrix lists must have same length")
        
        results = []
        
        # Queue all operations without synchronization
        for a, b in zip(matrices_a, matrices_b):
            result = torch.matmul(a, b)
            results.append(result)
        
        # Only synchronize if configured
        if self.device.config.synchronize_after_operations:
            self.device.synchronize()
        
        return results
    
    def parallel_model_inference(self, model: nn.Module, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Run model inference on multiple inputs in parallel
        """
        results = []
        
        with torch.no_grad():
            for input_tensor in inputs:
                # Each forward pass is queued on MPS
                output = model(input_tensor)
                results.append(output)
        
        # Optional synchronization
        if self.device.config.synchronize_after_operations:
            self.device.synchronize()
        
        return results


class MPSProfiler:
    """
    Profiling support for MPS operations
    """
    
    def __init__(self, device: MPSDevice):
        self.device = device
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @contextmanager
    def profile(self, name: str = "mps_operation"):
        """Profile MPS operations"""
        if not self.device.is_available:
            yield
            return
        
        if torch.mps.profiler.is_metal_capture_enabled():
            with torch.mps.profiler.metal_capture():
                with torch.mps.profiler.profile("interval"):
                    self.logger.debug(f"Profiling: {name}")
                    yield
        else:
            with torch.mps.profiler.profile("interval"):
                self.logger.debug(f"Profiling: {name}")
                yield
    
    def start_profiling(self):
        """Start continuous profiling"""
        if self.device.is_available:
            torch.mps.profiler.start("interval")
    
    def stop_profiling(self):
        """Stop continuous profiling"""
        if self.device.is_available:
            torch.mps.profiler.stop()