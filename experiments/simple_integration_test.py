#!/usr/bin/env python
"""
Simple Integration Test
Quick validation that architecture components work together
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import ray
import torch
import time

from thea_code_system.core import (
    EnhancedActor,
    EnhancedActorPool,
    TensorStore,
    ActorConfig
)


class SimpleActor(EnhancedActor):
    """Simple actor for testing"""
    
    async def process(self, x: float) -> float:
        """Process with PyTorch"""
        # All math with PyTorch
        tensor = self.scalar_ops.scalar(x)
        result = self.scalar_ops.mul(tensor, 2)
        result = self.scalar_ops.add(result, 1)
        return result.item()


async def main():
    """Run simple integration test"""
    print("\n" + "="*60)
    print("🎯 SIMPLE INTEGRATION TEST")
    print("="*60)
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    success = True
    
    try:
        # Test 1: Create enhanced pool
        print("\n📊 Test 1: Enhanced Actor Pool")
        
        config = ActorConfig(
            name="simple",
            num_cpus=1.0,
            num_gpus=0
        )
        
        pool = EnhancedActorPool(
            SimpleActor,
            num_actors=2,
            config=config
        )
        
        await pool.initialize()
        print("✅ Pool initialized")
        
        # Test 2: Process data
        print("\n📊 Test 2: Data Processing")
        
        data = [1, 2, 3, 4, 5]
        results = pool.map(
            lambda actor, x: actor.process.remote(x),
            data
        )
        
        expected = [x * 2 + 1 for x in data]
        if results == expected:
            print(f"✅ Results correct: {results}")
        else:
            print(f"❌ Results mismatch: {results} != {expected}")
            success = False
        
        # Test 3: Tensor store
        print("\n📊 Test 3: Tensor Store")
        
        store = TensorStore()
        tensor = torch.randn(10, 10)
        
        store.put("test", tensor)
        retrieved = store.get("test")
        
        if torch.allclose(retrieved, tensor):
            print("✅ Tensor store working")
        else:
            print("❌ Tensor store failed")
            success = False
        
        # Test 4: Metrics
        print("\n📊 Test 4: Metrics")
        
        print(f"   Tasks submitted: {pool.metrics.total_tasks_submitted}")
        print(f"   Tasks completed: {pool.metrics.total_tasks_completed}")
        print(f"   Latency: {pool.metrics.average_latency_ms:.2f}ms")
        
        if pool.metrics.total_tasks_completed > 0:
            print("✅ Metrics tracking working")
        else:
            print("❌ Metrics not tracking")
            success = False
        
        # Cleanup
        await pool.shutdown()
        print("\n✅ Clean shutdown")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    finally:
        ray.shutdown()
    
    # Summary
    print("\n" + "="*60)
    if success:
        print("🎉 Integration validated!")
        print("Architecture components working together:")
        print("  ✅ Enhanced Actor Pool")
        print("  ✅ PyTorch scalar operations") 
        print("  ✅ Tensor Store")
        print("  ✅ Metrics tracking")
    else:
        print("⚠️ Integration has issues")
    
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)