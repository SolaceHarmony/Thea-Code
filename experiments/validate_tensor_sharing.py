#!/usr/bin/env python
"""
Validate Tensor Sharing
Ensure tensor sharing between actors works correctly with proper device management
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ray
import torch
import time
import numpy as np
from typing import Any, Dict, Optional

from thea_code_system.core.enhanced_pool import (
    EnhancedActor, EnhancedActorPool, TensorStore
)
from thea_code_system.core.base import ActorConfig


class TensorWorker(EnhancedActor):
    """Actor that works with shared tensors"""
    
    async def process(self, data: Any) -> Any:
        """Required process method"""
        return data
    
    async def store_matrix(self, key: str, shape: tuple) -> bool:
        """Create and store a matrix"""
        # Create matrix on this actor's device
        matrix = torch.randn(*shape, device=self.device)
        self._tensor_cache[key] = matrix
        return True
    
    async def compute_with_shared(self, matrix_key: str, vector: torch.Tensor) -> torch.Tensor:
        """Compute using shared matrix"""
        # Get shared matrix
        matrix = self._tensor_cache.get(matrix_key)
        if matrix is None:
            raise ValueError(f"Matrix {matrix_key} not found in cache")
        
        # Ensure vector is on same device
        if not isinstance(vector, torch.Tensor):
            vector = torch.tensor(vector, device=self.device)
        elif vector.device != self.device:
            vector = vector.to(self.device)
        
        # Matrix-vector multiplication
        result = torch.matmul(matrix, vector)
        return result.cpu()  # Return on CPU for serialization
    
    async def aggregate_tensors(self, tensors: list) -> torch.Tensor:
        """Aggregate multiple tensors"""
        # Move all to same device and stack
        device_tensors = []
        for t in tensors:
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t, device=self.device)
            elif t.device != self.device:
                t = t.to(self.device)
            device_tensors.append(t)
        
        stacked = torch.stack(device_tensors)
        return stacked.mean(dim=0).cpu()


def test_basic_tensor_store():
    """Test basic TensorStore functionality"""
    print("\n📊 Test 1: Basic TensorStore")
    print("-" * 40)
    
    store = TensorStore()
    
    # Store tensors
    tensor1 = torch.randn(10, 10)
    tensor2 = torch.ones(5, 5) * 42
    
    ref1 = store.put("weights", tensor1)
    ref2 = store.put("biases", tensor2)
    
    print(f"✅ Stored 2 tensors in Ray object store")
    
    # Retrieve tensors
    retrieved1 = store.get("weights")
    retrieved2 = store.get("biases")
    
    assert torch.allclose(retrieved1, tensor1), "Tensor 1 mismatch"
    assert torch.allclose(retrieved2, tensor2), "Tensor 2 mismatch"
    
    print(f"✅ Retrieved tensors match originals")
    
    # List keys
    keys = store.list_keys()
    assert set(keys) == {"weights", "biases"}, f"Keys mismatch: {keys}"
    print(f"✅ Keys listed correctly: {keys}")
    
    # Delete tensor
    deleted = store.delete("biases")
    assert deleted, "Failed to delete"
    assert store.get("biases") is None, "Tensor not deleted"
    print(f"✅ Tensor deletion works")
    
    # Clear store
    store.clear()
    assert len(store.list_keys()) == 0, "Store not cleared"
    print(f"✅ Store cleared successfully")
    
    return True


def test_actor_tensor_sharing():
    """Test tensor sharing between actors"""
    print("\n📊 Test 2: Actor Tensor Sharing")
    print("-" * 40)
    
    # Initialize Ray if needed
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    try:
        # Create pool with tensor workers
        config = ActorConfig(
            name="tensor_pool",
            num_cpus=1.0,
            num_gpus=0  # CPU only for simplicity
        )
        
        pool = EnhancedActorPool(
            TensorWorker,
            num_actors=3,
            config=config
        )
        
        # Initialize actors
        init_tasks = []
        for actor in pool.actors:
            init_tasks.append(actor.initialize.remote())
        ray.get(init_tasks)
        
        print(f"✅ Created pool with {pool.num_actors} actors")
        
        # Have first actor create a matrix
        matrix_shape = (100, 50)
        actor0 = pool.actors[0]
        ray.get(actor0.store_matrix.remote("shared_weights", matrix_shape))
        print(f"✅ Actor 0 created matrix {matrix_shape}")
        
        # Get the matrix from actor 0
        matrix = ray.get(actor0.get_shared_tensor.remote("shared_weights"))
        
        # Broadcast to all actors
        broadcast_tasks = []
        for actor in pool.actors:
            broadcast_tasks.append(actor.share_tensor.remote("shared_weights", matrix))
        ray.get(broadcast_tasks)
        
        print(f"✅ Broadcasted matrix to all actors")
        
        # Verify all actors have it
        for i, actor in enumerate(pool.actors):
            has_matrix = ray.get(actor.get_shared_tensor.remote("shared_weights"))
            assert has_matrix is not None, f"Actor {i} missing matrix"
            assert has_matrix.shape == matrix_shape, f"Actor {i} matrix shape mismatch"
        
        print(f"✅ All actors have the shared matrix")
        
        # Test computation with shared matrix
        vector = torch.randn(50)
        
        compute_tasks = []
        for actor in pool.actors:
            compute_tasks.append(
                actor.compute_with_shared.remote("shared_weights", vector)
            )
        
        results = ray.get(compute_tasks)
        
        print(f"✅ All actors computed with shared matrix")
        
        # Results should be identical (same matrix, same vector)
        for i in range(1, len(results)):
            assert torch.allclose(results[0], results[i], rtol=1e-5), \
                f"Actor {i} result differs"
        
        print(f"✅ All actors produced identical results")
        
        # Cleanup
        pool.shutdown()
        
        return True
        
    except Exception as e:
        print(f"❌ Actor tensor sharing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tensor_aggregation():
    """Test aggregating tensors across actors"""
    print("\n📊 Test 3: Tensor Aggregation")
    print("-" * 40)
    
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    try:
        config = ActorConfig(
            name="aggregation_pool",
            num_cpus=1.0,
            num_gpus=0
        )
        
        pool = EnhancedActorPool(
            TensorWorker,
            num_actors=4,
            config=config
        )
        
        # Initialize actors
        init_tasks = []
        for actor in pool.actors:
            init_tasks.append(actor.initialize.remote())
        ray.get(init_tasks)
        
        print(f"✅ Created pool with {pool.num_actors} actors")
        
        # Each actor creates a tensor
        tensor_tasks = []
        for i, actor in enumerate(pool.actors):
            # Each actor creates a different tensor
            tensor = torch.ones(10) * (i + 1)  # [1,1,1...], [2,2,2...], etc
            tensor_tasks.append(actor.share_tensor.remote(f"tensor_{i}", tensor))
        
        ray.get(tensor_tasks)
        print(f"✅ Each actor created unique tensor")
        
        # Gather all tensors
        all_tensors = []
        for i, actor in enumerate(pool.actors):
            tensor = ray.get(actor.get_shared_tensor.remote(f"tensor_{i}"))
            all_tensors.append(tensor)
            print(f"   Actor {i} tensor: {tensor[:3].tolist()}...")
        
        # Have one actor aggregate all tensors
        aggregator = pool.actors[0]
        aggregated = ray.get(aggregator.aggregate_tensors.remote(all_tensors))
        
        # Expected: mean of [1,1,1...], [2,2,2...], [3,3,3...], [4,4,4...]
        expected_mean = (1 + 2 + 3 + 4) / 4  # = 2.5
        expected = torch.ones(10) * expected_mean
        
        assert torch.allclose(aggregated, expected), \
            f"Aggregation failed: {aggregated} != {expected}"
        
        print(f"✅ Aggregation correct: mean = {aggregated[0].item()}")
        
        # Cleanup
        pool.shutdown()
        
        return True
        
    except Exception as e:
        print(f"❌ Tensor aggregation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_large_tensor_performance():
    """Test performance with large tensors"""
    print("\n📊 Test 4: Large Tensor Performance")
    print("-" * 40)
    
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    try:
        store = TensorStore()
        
        # Test different tensor sizes
        sizes = [
            (100, 100),      # ~40KB
            (1000, 1000),    # ~4MB
            (5000, 5000),    # ~100MB
        ]
        
        for shape in sizes:
            tensor = torch.randn(*shape)
            size_mb = (tensor.element_size() * tensor.nelement()) / (1024 * 1024)
            
            # Store
            start = time.time()
            ref = store.put(f"tensor_{shape}", tensor)
            store_time = time.time() - start
            
            # Retrieve
            start = time.time()
            retrieved = store.get(f"tensor_{shape}")
            retrieve_time = time.time() - start
            
            print(f"   Shape {shape} ({size_mb:.1f}MB):")
            print(f"     Store:    {store_time*1000:.1f}ms")
            print(f"     Retrieve: {retrieve_time*1000:.1f}ms")
            
            assert torch.allclose(retrieved, tensor), "Tensor mismatch"
        
        print(f"✅ Large tensor handling works efficiently")
        
        # Clear
        store.clear()
        
        return True
        
    except Exception as e:
        print(f"❌ Large tensor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tensor sharing validation tests"""
    print("\n" + "="*60)
    print("🎯 TENSOR SHARING VALIDATION")
    print("="*60)
    
    results = {}
    
    # Test 1: Basic TensorStore
    results['TensorStore'] = test_basic_tensor_store()
    
    # Test 2: Actor tensor sharing
    results['Actor Sharing'] = test_actor_tensor_sharing()
    
    # Test 3: Tensor aggregation
    results['Aggregation'] = test_tensor_aggregation()
    
    # Test 4: Large tensor performance
    results['Large Tensors'] = test_large_tensor_performance()
    
    # Summary
    print("\n" + "="*60)
    print("📊 VALIDATION SUMMARY")
    print("="*60)
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name:20s}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 Tensor sharing validated successfully!")
        print("Ready to integrate into main architecture")
    else:
        print("\n⚠️ Some validations failed")
    
    # Cleanup
    if ray.is_initialized():
        ray.shutdown()
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)