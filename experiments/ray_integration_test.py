#!/usr/bin/env python
"""
Ray Integration Test
Tests what actually works with Ray actors
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import ray
import torch
import time
from typing import Dict, Any

from thea_code_system.core.base import BaseActor, BasePool, ActorConfig
from thea_code_system.core.scalars import ScalarOperations, AccumulatorBase


class TestRayActor(BaseActor):
    """Test Ray actor implementation"""
    
    async def initialize(self) -> None:
        """Initialize actor"""
        self.ops = ScalarOperations(self.device)
        self.accumulator = AccumulatorBase(self.device)
        self._initialized = True
        self.process_count = 0
        print(f"Actor {self.actor_id} initialized on {self.device}")
    
    async def process(self, data: float) -> Dict[str, Any]:
        """Process data"""
        # Use PyTorch scalars
        input_tensor = self.ops.scalar(data)
        result = self.ops.mul(input_tensor, 2)
        
        self.accumulator.add(result)
        self.process_count += 1
        
        return {
            'actor_id': self.actor_id,
            'input': data,
            'output': result.item(),
            'process_count': self.process_count
        }
    
    async def get_total_processed(self) -> int:
        """Get total items processed"""
        return self.process_count


async def test_ray_actors():
    """Test Ray actor creation and usage"""
    print("\n🔬 Testing Ray Actor Integration")
    print("-" * 40)
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    try:
        # Test 1: Create single Ray actor
        print("\n📊 Test 1: Single Ray Actor")
        
        config = ActorConfig(
            name="test_actor",
            num_gpus=0.25 if torch.cuda.is_available() else 0,
            num_cpus=1.0
        )
        
        # Create remote actor class
        RemoteActor = TestRayActor.as_remote(config)
        
        # Create actor instance
        actor = RemoteActor.remote("actor_0", config)
        
        # Initialize
        await actor.initialize.remote()
        print("✅ Actor created and initialized")
        
        # Process data
        result = await actor.process.remote(42.0)
        print(f"✅ Processed: {result}")
        
        # Get state
        count = await actor.get_total_processed.remote()
        print(f"✅ Total processed: {count}")
        
        # Test 2: Actor Pool with Ray
        print("\n📊 Test 2: Ray Actor Pool")
        
        # Create pool - this is where it might break
        pool = BasePool(TestRayActor, size=3, config=config)
        await pool.initialize()
        print(f"✅ Pool created with {pool.size} actors")
        
        # Process batch
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        results = await pool.map('process', data)
        print(f"✅ Processed {len(results)} items")
        
        # Check results
        for r in results[:3]:
            print(f"   Actor {r['actor_id']}: {r['input']} -> {r['output']}")
        
        # Pool health
        health = await pool.health_check()
        print(f"✅ Pool health: {health['healthy_workers']}/{health['pool_size']}")
        
        # Test 3: Parallel Processing
        print("\n📊 Test 3: Parallel Performance")
        
        # Large batch
        large_batch = list(range(100))
        
        start = time.time()
        results = await pool.map('process', large_batch)
        elapsed = time.time() - start
        
        print(f"✅ Processed {len(results)} items in {elapsed:.3f}s")
        print(f"   Throughput: {len(results)/elapsed:.1f} items/sec")
        
        # Verify all items processed
        outputs = [r['output'] for r in results]
        expected = [i * 2 for i in large_batch]
        
        if outputs == expected:
            print("✅ All results correct")
        else:
            print("❌ Results don't match expected")
        
        # Cleanup
        await pool.shutdown()
        
        return True
        
    except Exception as e:
        print(f"❌ Ray integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        ray.shutdown()


async def test_ray_specific_features():
    """Test Ray-specific features that might not work"""
    print("\n🔬 Testing Ray-Specific Features")
    print("-" * 40)
    
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    try:
        # Test Ray futures handling
        print("\n📊 Testing Ray Futures")
        
        @ray.remote
        class SimpleActor:
            def __init__(self):
                self.value = 0
            
            def increment(self):
                self.value += 1
                return self.value
            
            async def async_increment(self):
                await asyncio.sleep(0.001)
                self.value += 1
                return self.value
        
        actor = SimpleActor.remote()
        
        # Test sync method
        future = actor.increment.remote()
        result = ray.get(future)
        print(f"✅ Sync method: {result}")
        
        # Test async method
        future = actor.async_increment.remote()
        result = await future  # This might not work
        print(f"✅ Async method: {result}")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Ray-specific feature failed: {e}")
        return False
    
    finally:
        ray.shutdown()


async def main():
    """Run all Ray integration tests"""
    print("\n" + "="*60)
    print("🎯 RAY INTEGRATION TEST")
    print("="*60)
    
    results = {}
    
    # Test basic Ray actors
    results['Ray Actors'] = await test_ray_actors()
    
    # Test Ray-specific features
    results['Ray Features'] = await test_ray_specific_features()
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name:20s}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 Ray integration working!")
    else:
        print("\n⚠️ Ray integration has issues")
        print("\nKnown issues:")
        print("  - Async/await with Ray futures needs work")
        print("  - Pool initialization with Ray actors untested")
        print("  - Actor lifecycle management incomplete")
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)