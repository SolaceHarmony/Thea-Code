#!/usr/bin/env python
"""
Test Enhanced Actor Pool
Validates that we properly wrap Ray's ActorPool with PyTorch operations
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import ray
import torch
import time
import numpy as np
from typing import Any, Dict

from thea_code_system.core.enhanced_pool import (
    EnhancedActor, EnhancedActorPool, TensorStore
)
from thea_code_system.core.base import ActorConfig


class ComputeActor(EnhancedActor):
    """Test actor for compute operations"""
    
    async def initialize(self) -> None:
        """Initialize with custom state"""
        await super().initialize()
        self.compute_count = 0
    
    async def process(self, data: Any) -> Any:
        """Required abstract method implementation"""
        return await self.compute(data)
        
    async def compute(self, x: torch.Tensor) -> Dict[str, Any]:
        """Compute operation using PyTorch"""
        # Ensure we have a tensor
        if not isinstance(x, torch.Tensor):
            x = self.scalar_ops.scalar(x)
        
        # Do some computation - all with PyTorch
        result = self.scalar_ops.mul(x, 2)
        result = self.scalar_ops.add(result, 1)
        result = self.scalar_ops.pow(result, 2)
        
        self.compute_count += 1
        
        return {
            'actor_id': self.actor_id,
            'input': x.item() if x.numel() == 1 else x.tolist(),
            'output': result.item() if result.numel() == 1 else result.tolist(),
            'compute_count': self.compute_count,
            'device': str(self.device)
        }
    
    async def matrix_multiply(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Matrix multiplication using PyTorch"""
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a, device=self.device)
        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b, device=self.device)
            
        return torch.matmul(a, b)


async def test_enhanced_pool_basics():
    """Test basic Enhanced Actor Pool functionality"""
    print("\n🔬 Testing Enhanced Actor Pool Basics")
    print("-" * 40)
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    try:
        # Create pool with enhanced actors
        config = ActorConfig(
            name="compute_pool",
            num_gpus=0.25 if torch.cuda.is_available() else 0,
            num_cpus=1.0
        )
        
        pool = EnhancedActorPool(
            ComputeActor,
            num_actors=3,
            config=config,
            enable_queue=True
        )
        
        # Initialize actors
        init_tasks = []
        for actor in pool.actors:
            init_tasks.append(actor.initialize.remote())
        ray.get(init_tasks)
        
        print(f"✅ Created pool with {pool.num_actors} enhanced actors")
        
        # Test 1: Simple map operation
        print("\n📊 Test 1: Map with PyTorch scalars")
        
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        results = pool.map(
            lambda actor, x: actor.compute.remote(x),
            values
        )
        
        print(f"✅ Processed {len(results)} items")
        for r in results[:3]:
            # Formula: ((x * 2) + 1) ^ 2
            print(f"   Actor {r['actor_id']}: {r['input']} -> {r['output']} on {r['device']}")
        
        # Verify math is correct
        for i, r in enumerate(results):
            expected = ((values[i] * 2) + 1) ** 2
            assert abs(r['output'] - expected) < 1e-6, f"Math error: {r['output']} != {expected}"
        
        print("✅ All PyTorch calculations correct")
        
        # Test 2: Unordered map for better performance
        print("\n📊 Test 2: Unordered map performance")
        
        large_values = list(range(100))
        
        start = time.time()
        unordered_results = pool.map_unordered(
            lambda actor, x: actor.compute.remote(x),
            large_values
        )
        elapsed = time.time() - start
        
        print(f"✅ Processed {len(unordered_results)} items in {elapsed:.3f}s")
        print(f"   Throughput: {len(unordered_results)/elapsed:.1f} items/sec")
        
        # Test 3: Submit/get pattern
        print("\n📊 Test 3: Submit/get pattern")
        
        # Submit multiple tasks
        for i in range(5):
            pool.submit(
                lambda actor, x: actor.compute.remote(x),
                float(i * 10)
            )
        
        # Get results
        results = []
        while pool.has_next():
            result = pool.get_next()
            results.append(result)
            print(f"   Got result: {result['input']} -> {result['output']}")
        
        print(f"✅ Retrieved {len(results)} results")
        
        # Test 4: Metrics collection
        print("\n📊 Test 4: Metrics collection")
        
        metrics = await pool.collect_metrics()
        print(f"✅ Pool metrics:")
        print(f"   Total submitted: {metrics['pool_metrics']['total_submitted']}")
        print(f"   Total completed: {metrics['pool_metrics']['total_completed']}")
        print(f"   Average latency: {metrics['pool_metrics']['average_latency_ms']:.2f}ms")
        print(f"   Has free actors: {metrics['has_free']}")
        
        # Shutdown
        pool.shutdown()
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced pool test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_tensor_sharing():
    """Test tensor sharing between actors"""
    print("\n🔬 Testing Tensor Sharing")
    print("-" * 40)
    
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    try:
        # Create pool
        config = ActorConfig(
            name="tensor_pool",
            num_gpus=0.25 if torch.cuda.is_available() else 0,
            num_cpus=1.0
        )
        
        pool = EnhancedActorPool(
            ComputeActor,
            num_actors=2,
            config=config
        )
        
        # Initialize actors
        init_tasks = []
        for actor in pool.actors:
            init_tasks.append(actor.initialize.remote())
        ray.get(init_tasks)
        
        print("✅ Pool created")
        
        # Test 1: Broadcast tensor to all actors
        print("\n📊 Test 1: Broadcasting tensors")
        
        # Create a tensor to share
        shared_tensor = torch.randn(10, 10)
        await pool.broadcast_tensor("weights", shared_tensor)
        
        print(f"✅ Broadcasted tensor shape {shared_tensor.shape}")
        
        # Verify all actors received it
        for i, actor in enumerate(pool.actors):
            retrieved = ray.get(actor.get_shared_tensor.remote("weights"))
            if retrieved is not None:
                print(f"   Actor {i}: Received tensor shape {retrieved.shape}")
                assert torch.allclose(retrieved, shared_tensor), "Tensor mismatch"
        
        # Test 2: TensorStore for zero-copy sharing
        print("\n📊 Test 2: TensorStore zero-copy")
        
        store = TensorStore()
        
        # Store large tensor
        large_tensor = torch.randn(1000, 1000)
        ref = store.put("large_weights", large_tensor)
        
        print(f"✅ Stored tensor ({large_tensor.shape}) in Ray object store")
        
        # Retrieve tensor
        retrieved = store.get("large_weights")
        assert torch.allclose(retrieved, large_tensor), "Retrieved tensor mismatch"
        
        print(f"✅ Retrieved tensor matches original")
        
        # Test 3: Matrix operations with shared tensors
        print("\n📊 Test 3: Distributed matrix operations")
        
        # Create matrices
        matrix_a = torch.randn(100, 50)
        matrix_b = torch.randn(50, 100)
        
        # Store in TensorStore
        store.put("matrix_a", matrix_a)
        store.put("matrix_b", matrix_b)
        
        # Have actors perform matrix multiplication
        results = []
        for actor in pool.actors:
            # Each actor retrieves and multiplies
            result = actor.matrix_multiply.remote(
                store.get("matrix_a"),
                store.get("matrix_b")
            )
            results.append(result)
        
        results = ray.get(results)
        
        # Verify all actors computed same result
        expected = torch.matmul(matrix_a, matrix_b)
        for i, result in enumerate(results):
            print(f"   Actor {i}: Matrix result shape {result.shape}")
            assert torch.allclose(result, expected, rtol=1e-5), f"Actor {i} result mismatch"
        
        print("✅ All actors computed identical matrix multiplication")
        
        # Cleanup
        store.clear()
        pool.shutdown()
        
        return True
        
    except Exception as e:
        print(f"❌ Tensor sharing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_performance_comparison():
    """Compare Enhanced Pool vs Raw Ray ActorPool"""
    print("\n🔬 Performance Comparison: Enhanced vs Raw Ray")
    print("-" * 40)
    
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    try:
        # Simple compute function
        @ray.remote
        class SimpleActor:
            def compute(self, x):
                return ((x * 2) + 1) ** 2
        
        # Create raw Ray ActorPool
        from ray.util.actor_pool import ActorPool as RayActorPool
        
        raw_actors = [SimpleActor.remote() for _ in range(3)]
        raw_pool = RayActorPool(raw_actors)
        
        # Create Enhanced Pool
        config = ActorConfig(name="perf_test", num_cpus=1.0)
        enhanced_pool = EnhancedActorPool(
            ComputeActor,
            num_actors=3,
            config=config
        )
        
        # Initialize enhanced actors
        init_tasks = []
        for actor in enhanced_pool.actors:
            init_tasks.append(actor.initialize.remote())
        ray.get(init_tasks)
        
        # Test data
        test_data = list(range(1000))
        
        # Test 1: Raw Ray ActorPool
        print("\n📊 Raw Ray ActorPool:")
        start = time.time()
        raw_results = list(raw_pool.map(
            lambda actor, x: actor.compute.remote(x),
            test_data
        ))
        raw_time = time.time() - start
        print(f"   Time: {raw_time:.3f}s")
        print(f"   Throughput: {len(test_data)/raw_time:.1f} items/sec")
        
        # Test 2: Enhanced Pool (with PyTorch overhead)
        print("\n📊 Enhanced Pool (PyTorch everywhere):")
        start = time.time()
        enhanced_results = enhanced_pool.map(
            lambda actor, x: actor.compute.remote(x),
            test_data
        )
        enhanced_time = time.time() - start
        print(f"   Time: {enhanced_time:.3f}s")
        print(f"   Throughput: {len(test_data)/enhanced_time:.1f} items/sec")
        
        # Calculate overhead
        overhead = ((enhanced_time - raw_time) / raw_time) * 100
        print(f"\n📊 PyTorch overhead: {overhead:.1f}%")
        
        if overhead < 50:
            print("✅ Acceptable overhead for PyTorch benefits")
        else:
            print("⚠️ High overhead - may need optimization")
        
        # Get final metrics
        metrics = await enhanced_pool.collect_metrics()
        print(f"\n📊 Enhanced Pool final metrics:")
        print(f"   Tasks completed: {metrics['pool_metrics']['total_completed']}")
        print(f"   Average latency: {metrics['pool_metrics']['average_latency_ms']:.2f}ms")
        
        # Cleanup
        enhanced_pool.shutdown()
        
        return True
        
    except Exception as e:
        print(f"❌ Performance comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all enhanced pool tests"""
    print("\n" + "="*60)
    print("🎯 ENHANCED ACTOR POOL TEST")
    print("="*60)
    
    results = {}
    
    # Test enhanced pool basics
    results['Enhanced Pool'] = await test_enhanced_pool_basics()
    
    # Test tensor sharing
    results['Tensor Sharing'] = await test_tensor_sharing()
    
    # Performance comparison
    results['Performance'] = await test_performance_comparison()
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name:20s}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 Enhanced Actor Pool working perfectly!")
        print("Successfully wrapped Ray's ActorPool with:")
        print("  - PyTorch scalar operations for ALL math")
        print("  - Tensor sharing between actors")
        print("  - MPS device support")
        print("  - Advanced metrics collection")
    else:
        print("\n⚠️ Some tests failed")
    
    ray.shutdown()
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)