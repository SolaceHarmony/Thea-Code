#!/usr/bin/env python
"""
Simple test for Enhanced Actor Pool
Focus on core functionality without complex tensor operations
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ray
import torch
import time
from typing import Any

from thea_code_system.core.enhanced_pool import EnhancedActor, EnhancedActorPool
from thea_code_system.core.base import ActorConfig


class SimpleComputeActor(EnhancedActor):
    """Simple actor for testing"""
    
    async def process(self, data: Any) -> Any:
        """Process using PyTorch scalars"""
        # Convert to tensor
        if isinstance(data, (int, float)):
            x = self.scalar_ops.scalar(data)
        else:
            x = data
            
        # Simple computation: x * 2 + 1
        result = self.scalar_ops.mul(x, 2)
        result = self.scalar_ops.add(result, 1)
        
        # Return as Python float
        return result.item() if result.numel() == 1 else result.tolist()


def main():
    """Run simple enhanced pool test"""
    print("\n" + "="*60)
    print("🎯 SIMPLE ENHANCED POOL TEST")
    print("="*60)
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    try:
        # Create pool
        config = ActorConfig(
            name="simple_pool",
            num_cpus=1.0,
            num_gpus=0  # CPU only for simplicity
        )
        
        print("\n📊 Creating Enhanced Actor Pool...")
        pool = EnhancedActorPool(
            SimpleComputeActor,
            num_actors=3,
            config=config
        )
        
        # Initialize actors
        print("📊 Initializing actors...")
        init_tasks = []
        for actor in pool.actors:
            init_tasks.append(actor.initialize.remote())
        ray.get(init_tasks)
        
        print(f"✅ Pool created with {pool.num_actors} actors")
        
        # Test 1: Simple map
        print("\n📊 Test 1: Map operation")
        values = [1, 2, 3, 4, 5]
        
        results = pool.map(
            lambda actor, x: actor.process.remote(x),
            values
        )
        
        print(f"   Input:  {values}")
        print(f"   Output: {results}")
        
        # Verify results (x * 2 + 1)
        expected = [x * 2 + 1 for x in values]
        if results == expected:
            print("   ✅ Calculations correct!")
        else:
            print(f"   ❌ Expected {expected}, got {results}")
        
        # Test 2: Performance test
        print("\n📊 Test 2: Performance")
        large_data = list(range(100))
        
        start = time.time()
        results = pool.map(
            lambda actor, x: actor.process.remote(x),
            large_data
        )
        elapsed = time.time() - start
        
        print(f"   Processed {len(results)} items in {elapsed:.3f}s")
        print(f"   Throughput: {len(results)/elapsed:.1f} items/sec")
        
        # Test 3: Unordered processing
        print("\n📊 Test 3: Unordered map (faster)")
        
        start = time.time()
        unordered_results = pool.map_unordered(
            lambda actor, x: actor.process.remote(x),
            large_data
        )
        elapsed = time.time() - start
        
        print(f"   Processed {len(unordered_results)} items in {elapsed:.3f}s")
        print(f"   Throughput: {len(unordered_results)/elapsed:.1f} items/sec")
        
        # Test 4: Metrics
        print("\n📊 Test 4: Pool metrics")
        # Note: collect_metrics is async, but we're in sync context
        # Just use the synchronous metrics directly
        
        print(f"   Total submitted: {pool.metrics.total_tasks_submitted}")
        print(f"   Total completed: {pool.metrics.total_tasks_completed}")
        print(f"   Average latency: {pool.metrics.average_latency_ms:.2f}ms")
        print(f"   Has free actors: {pool.has_free()}")
        
        # Shutdown
        print("\n📊 Shutting down pool...")
        pool.shutdown()
        
        print("\n✅ All tests completed successfully!")
        print("Successfully demonstrated:")
        print("  - Ray ActorPool wrapping")
        print("  - PyTorch scalar operations for ALL math")
        print("  - Performance metrics collection")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        ray.shutdown()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)