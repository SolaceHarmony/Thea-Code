#!/usr/bin/env python
"""
Tensor Store Actor
Handles out-of-band tensor communication between actors

This enables efficient distributed processing without serializing large tensors
"""

import ray
import torch
import numpy as np
from typing import Dict, Optional, Tuple, Any
import time
import gc


@ray.remote
class TensorStoreActor:
    """
    Centralized tensor storage for efficient actor communication
    
    Benefits:
    - Avoid serializing large tensors
    - Enable zero-copy sharing when possible
    - Automatic cleanup of unused tensors
    - Memory usage monitoring
    """
    
    def __init__(self, max_size_gb: float = 4.0):
        self.tensors = {}
        self.tensor_metadata = {}
        self.tensor_id_counter = 0
        self.max_size_bytes = int(max_size_gb * 1024**3)
        
        print(f"🗄️  TensorStore initialized with {max_size_gb}GB limit")
    
    def _generate_tensor_id(self) -> str:
        """Generate unique tensor ID"""
        self.tensor_id_counter += 1
        return f"tensor_{self.tensor_id_counter}_{int(time.time()*1000)}"
    
    def _get_tensor_size(self, tensor: torch.Tensor) -> int:
        """Get tensor size in bytes"""
        return tensor.numel() * tensor.element_size()
    
    def _cleanup_old_tensors(self):
        """Remove old tensors if memory usage is high"""
        
        total_size = sum(
            meta['size_bytes'] 
            for meta in self.tensor_metadata.values()
        )
        
        if total_size > self.max_size_bytes:
            # Sort by access time, remove oldest
            sorted_tensors = sorted(
                self.tensor_metadata.items(),
                key=lambda x: x[1]['last_access']
            )
            
            # Remove oldest 25%
            to_remove = len(sorted_tensors) // 4
            for tensor_id, _ in sorted_tensors[:to_remove]:
                self.delete_tensor(tensor_id)
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    async def store_tensor(self, tensor: torch.Tensor, metadata: Optional[Dict] = None) -> str:
        """
        Store tensor and return ID
        
        Args:
            tensor: PyTorch tensor to store
            metadata: Optional metadata about the tensor
            
        Returns:
            tensor_id: Unique identifier for retrieval
        """
        
        tensor_id = self._generate_tensor_id()
        tensor_size = self._get_tensor_size(tensor)
        
        # Store tensor (move to CPU to avoid GPU memory issues)
        stored_tensor = tensor.cpu().detach().clone()
        
        self.tensors[tensor_id] = stored_tensor
        self.tensor_metadata[tensor_id] = {
            'shape': tuple(tensor.shape),
            'dtype': str(tensor.dtype),
            'device': str(tensor.device),
            'size_bytes': tensor_size,
            'created_at': time.time(),
            'last_access': time.time(),
            'access_count': 0,
            'metadata': metadata or {}
        }
        
        # Cleanup if needed
        self._cleanup_old_tensors()
        
        return tensor_id
    
    async def retrieve_tensor(self, 
                            tensor_id: str, 
                            target_device: Optional[str] = None) -> Optional[torch.Tensor]:
        """
        Retrieve tensor by ID
        
        Args:
            tensor_id: Tensor identifier
            target_device: Device to move tensor to
            
        Returns:
            tensor: Retrieved tensor or None if not found
        """
        
        if tensor_id not in self.tensors:
            return None
        
        # Update access info
        meta = self.tensor_metadata[tensor_id]
        meta['last_access'] = time.time()
        meta['access_count'] += 1
        
        # Get tensor
        tensor = self.tensors[tensor_id]
        
        # Move to target device if specified
        if target_device:
            try:
                device = torch.device(target_device)
                tensor = tensor.to(device)
            except Exception as e:
                print(f"⚠️  Failed to move tensor to {target_device}: {e}")
        
        return tensor
    
    async def delete_tensor(self, tensor_id: str) -> bool:
        """Delete tensor from storage"""
        
        if tensor_id in self.tensors:
            del self.tensors[tensor_id]
            del self.tensor_metadata[tensor_id]
            return True
        
        return False
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        
        total_tensors = len(self.tensors)
        total_size = sum(meta['size_bytes'] for meta in self.tensor_metadata.values())
        
        if total_tensors > 0:
            avg_size = total_size / total_tensors
            most_accessed = max(
                self.tensor_metadata.items(),
                key=lambda x: x[1]['access_count']
            )
        else:
            avg_size = 0
            most_accessed = None
        
        return {
            'total_tensors': total_tensors,
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024**2),
            'avg_tensor_size_bytes': avg_size,
            'memory_usage_ratio': total_size / self.max_size_bytes,
            'most_accessed_tensor': most_accessed[0] if most_accessed else None,
            'max_access_count': most_accessed[1]['access_count'] if most_accessed else 0
        }
    
    async def list_tensors(self) -> Dict[str, Dict]:
        """List all stored tensors with metadata"""
        
        return {
            tensor_id: {
                'shape': meta['shape'],
                'dtype': meta['dtype'],
                'size_mb': meta['size_bytes'] / (1024**2),
                'access_count': meta['access_count'],
                'age_seconds': time.time() - meta['created_at']
            }
            for tensor_id, meta in self.tensor_metadata.items()
        }
    
    async def cleanup_all(self):
        """Clear all tensors (use with caution!)"""
        
        self.tensors.clear()
        self.tensor_metadata.clear()
        
        # Force cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("🧹 TensorStore cleared")


# Utility functions for working with TensorStore
class TensorStoreClient:
    """Client wrapper for easier TensorStore usage"""
    
    def __init__(self, store_actor: TensorStoreActor):
        self.store = store_actor
    
    async def put(self, tensor: torch.Tensor, **metadata) -> str:
        """Store tensor with metadata"""
        return await self.store.store_tensor.remote(tensor, metadata)
    
    async def get(self, tensor_id: str, device: Optional[str] = None) -> Optional[torch.Tensor]:
        """Retrieve tensor"""
        return await self.store.retrieve_tensor.remote(tensor_id, device)
    
    async def delete(self, tensor_id: str) -> bool:
        """Delete tensor"""
        return await self.store.delete_tensor.remote(tensor_id)
    
    async def stats(self) -> Dict[str, Any]:
        """Get storage stats"""
        return await self.store.get_storage_stats.remote()


# Export for easy importing
__all__ = ['TensorStoreActor', 'TensorStoreClient']