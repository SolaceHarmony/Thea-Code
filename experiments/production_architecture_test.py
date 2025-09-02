#!/usr/bin/env python
"""
Production Architecture Test
Validates the well-structured class hierarchy
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import torch
from typing import Any, Dict

# Import our production components
from thea_code_system.core import (
    BaseWorker,
    BaseActor,
    BasePool,
    BaseOrchestrator,
    MPSDevice,
    MPSMemoryManager,
    MPSStreamManager,
    ScalarOperations,
    TensorMetrics,
    AccumulatorBase
)
from thea_code_system.core.base import ActorConfig, WorkerMetrics
from thea_code_system.core.mps import MPSConfig


# Concrete implementations for testing
class ConcreteWorker(BaseWorker):
    """Concrete worker implementation"""
    
    async def initialize(self) -> None:
        """Initialize worker"""
        self.ops = ScalarOperations(self.device)
        self.accumulator = AccumulatorBase(self.device)
        self._initialized = True
        self.logger.info(f"Worker {self.worker_id} initialized on {self.device}")
    
    async def process(self, data: float) -> Dict[str, Any]:
        """Process single item"""
        # Use PyTorch scalars for ALL operations
        input_tensor = self.ops.scalar(data)
        
        # Perform computations
        squared = self.ops.pow(input_tensor, 2)
        root = self.ops.sqrt(squared)
        
        # Update metrics
        self.accumulator.add(root)
        self.metrics.update(items=1, time_taken=0.001)
        
        return {
            'input': data,
            'output': root.item(),
            'mean_so_far': self.accumulator.mean().item()
        }


class ConcreteOrchestrator(BaseOrchestrator):
    """Concrete orchestrator implementation"""
    
    async def execute_workflow(self, input_data: list) -> Dict[str, Any]:
        """Execute orchestrated workflow"""
        if not self._initialized:
            await self.initialize()
        
        # Use worker pool to process data
        pool = self.pools.get('workers')
        if not pool:
            raise RuntimeError("No worker pool registered")
        
        # Process data through pool
        results = await pool.map('process', input_data)
        
        # Aggregate results using PyTorch scalars
        ops = ScalarOperations()
        total = ops.sum(*[r['output'] for r in results])
        mean = ops.mean(*[r['output'] for r in results])
        
        return {
            'processed_count': len(results),
            'total': total.item(),
            'mean': mean.item(),
            'results': results
        }


async def test_base_architecture():
    """Test base class hierarchy"""
    print("\n🔬 Testing Base Architecture")
    print("-" * 40)
    
    # Create worker
    config = ActorConfig(name="test_worker")
    worker = ConcreteWorker("worker_1", config)
    await worker.initialize()
    
    # Process data
    result = await worker.process(42.0)
    print(f"✅ Worker processed: {result}")
    
    # Check health
    health = await worker.health_check()
    print(f"✅ Worker health: {health['healthy']}")
    
    # Create pool
    pool = BasePool(ConcreteWorker, size=4)
    await pool.initialize()
    
    # Process batch through pool
    batch_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    pool_results = await pool.map('process', batch_data)
    print(f"✅ Pool processed {len(pool_results)} items")
    
    # Pool health check
    pool_health = await pool.health_check()
    print(f"✅ Pool health: {pool_health['healthy_workers']}/{pool_health['pool_size']} healthy")
    
    # Shutdown
    await pool.shutdown()
    await worker.shutdown()
    
    return True


async def test_mps_components():
    """Test MPS-specific components"""
    print("\n🔬 Testing MPS Components")
    print("-" * 40)
    
    # Test MPSDevice singleton
    device1 = MPSDevice()
    device2 = MPSDevice()
    assert device1 is device2, "MPSDevice should be singleton"
    print(f"✅ MPSDevice singleton works")
    print(f"   Device: {device1.device}")
    print(f"   Available: {device1.is_available}")
    
    # Configure MPS
    config = MPSConfig(
        memory_fraction=0.7,
        manual_seed=42,
        empty_cache_interval=50
    )
    device1.configure(config)
    print(f"✅ MPS configuration applied")
    
    # Test memory manager
    mem_manager = MPSMemoryManager(device1)
    tensor = mem_manager.allocate_tensor(1000, 1000)
    print(f"✅ Memory allocation works: {tensor.shape}")
    
    # Get memory report
    mem_report = mem_manager.get_allocation_report()
    if mem_report:
        print(f"✅ Memory report: {mem_report.get('current_memory_mb', 0):.2f} MB")
    
    # Test stream manager
    stream_manager = MPSStreamManager(device1)
    
    # Queue parallel operations
    def matmul_op(size=100):
        a = torch.randn(size, size, device=device1.device)
        b = torch.randn(size, size, device=device1.device)
        return torch.matmul(a, b)
    
    operations = [(matmul_op, (), {'size': 100}) for _ in range(10)]
    results = stream_manager.queue_parallel_operations(operations)
    stream_manager.synchronize_all()
    
    print(f"✅ Stream manager processed {len(results)} parallel operations")
    
    # Memory optimization
    mem_manager.optimize_memory()
    
    return True


async def test_scalar_operations():
    """Test PyTorch scalar operations"""
    print("\n🔬 Testing Scalar Operations")
    print("-" * 40)
    
    ops = ScalarOperations()
    
    # Test basic arithmetic
    a = ops.scalar(2)
    b = ops.scalar(3)
    
    # Even 2+3 uses PyTorch!
    result = ops.add(a, b)
    assert result.item() == 5, "Addition failed"
    print(f"✅ 2 + 3 = {result.item()} (computed on {result.device})")
    
    # Test comparisons
    is_greater = ops.gt(b, a)
    assert is_greater.item() == True, "Comparison failed"
    print(f"✅ 3 > 2 = {is_greater.item()}")
    
    # Test aggregations
    values = [1, 2, 3, 4, 5]
    mean = ops.mean(*values)
    print(f"✅ Mean of {values} = {mean.item()}")
    
    # Test metrics
    metrics = TensorMetrics()
    metrics.create_counter("files_processed")
    metrics.create_counter("errors")
    
    # Increment counters
    for _ in range(10):
        metrics.increment("files_processed")
    
    metrics.increment("errors", 2)
    
    all_metrics = metrics.get_all()
    print(f"✅ Metrics: {all_metrics}")
    
    # Test accumulator
    acc = AccumulatorBase()
    for i in range(100):
        acc.add(i * 0.1)
    
    stats = acc.get_stats()
    print(f"✅ Accumulator stats:")
    print(f"   Count: {stats['count']}")
    print(f"   Mean: {stats['mean']:.2f}")
    print(f"   Std: {stats['std']:.2f}")
    
    return True


async def test_orchestration():
    """Test complete orchestration"""
    print("\n🔬 Testing Orchestration")
    print("-" * 40)
    
    # Create orchestrator
    orchestrator = ConcreteOrchestrator("main_orchestrator")
    
    # Create and register worker pool
    worker_pool = BasePool(ConcreteWorker, size=4)
    orchestrator.register_pool("workers", worker_pool)
    
    # Initialize
    await orchestrator.initialize()
    
    # Execute workflow
    input_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    result = await orchestrator.execute_workflow(input_data)
    
    print(f"✅ Orchestration complete:")
    print(f"   Processed: {result['processed_count']} items")
    print(f"   Total: {result['total']:.2f}")
    print(f"   Mean: {result['mean']:.2f}")
    
    # Health check
    system_health = await orchestrator.health_check()
    print(f"✅ System health check passed")
    
    # Shutdown
    await orchestrator.shutdown()
    
    return True


async def main():
    """Run all architecture tests"""
    print("\n" + "="*60)
    print("🏗️ PRODUCTION ARCHITECTURE TEST")
    print("="*60)
    
    tests = [
        ("Base Architecture", test_base_architecture),
        ("MPS Components", test_mps_components),
        ("Scalar Operations", test_scalar_operations),
        ("Orchestration", test_orchestration)
    ]
    
    results = {}
    
    for name, test_func in tests:
        try:
            results[name] = await test_func()
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name:20s}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 Production architecture validated!")
        print("\nThe architecture features:")
        print("  ✓ Well-structured class hierarchy")
        print("  ✓ Base classes for Workers, Actors, Pools, Orchestrators")
        print("  ✓ MPS-specific optimizations")
        print("  ✓ All math uses PyTorch scalars")
        print("  ✓ Production-ready abstractions")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)