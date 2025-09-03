#!/usr/bin/env python
"""
MPS Validation Suite
Testing MPS capabilities before promoting to production architecture
"""

import torch
import torch.mps
import torch.nn as nn
import time
from typing import Dict, List, Optional
import asyncio

# Validate MPS availability
if not torch.backends.mps.is_available():
    print("⚠️ MPS not available, using CPU fallback")
    device = torch.device("cpu")
else:
    device = torch.device("mps")
    print(f"✅ MPS Device Available")
    print(f"  Device count: {torch.mps.device_count()}")
    print(f"  Current memory: {torch.mps.current_allocated_memory() / 1024**2:.2f} MB")


def test_basic_mps_operations():
    """Test basic MPS operations work correctly"""
    print("\n🔬 Testing Basic MPS Operations")
    print("-" * 40)
    
    # Test 1: Tensor creation
    try:
        tensor = torch.randn(1000, 1000, device=device)
        print("✅ Tensor creation works")
    except Exception as e:
        print(f"❌ Tensor creation failed: {e}")
        return False
    
    # Test 2: Matrix multiplication
    try:
        result = torch.matmul(tensor, tensor)
        print("✅ Matrix multiplication works")
    except Exception as e:
        print(f"❌ Matrix multiplication failed: {e}")
        return False
    
    # Test 3: Synchronization
    try:
        torch.mps.synchronize()
        print("✅ MPS synchronization works")
    except Exception as e:
        print(f"❌ MPS synchronization failed: {e}")
        return False
    
    # Test 4: Memory management
    try:
        initial_mem = torch.mps.current_allocated_memory()
        large_tensor = torch.randn(5000, 5000, device=device)
        peak_mem = torch.mps.current_allocated_memory()
        del large_tensor
        torch.mps.empty_cache()
        final_mem = torch.mps.current_allocated_memory()
        print(f"✅ Memory management works")
        print(f"   Initial: {initial_mem/1024**2:.2f} MB")
        print(f"   Peak: {peak_mem/1024**2:.2f} MB")
        print(f"   After cleanup: {final_mem/1024**2:.2f} MB")
    except Exception as e:
        print(f"❌ Memory management failed: {e}")
        return False
    
    return True


def test_parallel_operations():
    """Test parallel operation execution without explicit sync"""
    print("\n🔬 Testing Parallel Operations")
    print("-" * 40)
    
    size = 1000
    num_ops = 20
    
    # Sequential with sync after each
    start = time.time()
    for _ in range(num_ops):
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        c = torch.matmul(a, b)
        torch.mps.synchronize()  # Wait after each
    seq_time = time.time() - start
    
    # Parallel without sync
    start = time.time()
    operations = []
    for _ in range(num_ops):
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        c = torch.matmul(a, b)
        operations.append(c)
    torch.mps.synchronize()  # Wait once at end
    par_time = time.time() - start
    
    speedup = seq_time / par_time
    print(f"✅ Parallel execution works")
    print(f"   Sequential: {seq_time:.3f}s")
    print(f"   Parallel: {par_time:.3f}s")
    print(f"   Speedup: {speedup:.2f}x")
    
    return speedup > 1.2  # At least 20% speedup


def test_model_compilation():
    """Test model compilation for MPS"""
    print("\n🔬 Testing Model Compilation")
    print("-" * 40)
    
    # Create simple model
    model = nn.Sequential(
        nn.Linear(768, 1024),
        nn.ReLU(),
        nn.Linear(1024, 768)
    ).to(device)
    
    # Test without compilation
    input_tensor = torch.randn(100, 768, device=device)
    
    start = time.time()
    for _ in range(100):
        output = model(input_tensor)
        torch.mps.synchronize()
    uncompiled_time = time.time() - start
    
    # Test with compilation
    try:
        compiled_model = torch.compile(model, backend="aot_eager")
        
        start = time.time()
        for _ in range(100):
            output = compiled_model(input_tensor)
            torch.mps.synchronize()
        compiled_time = time.time() - start
        
        speedup = uncompiled_time / compiled_time
        print(f"✅ Model compilation works")
        print(f"   Uncompiled: {uncompiled_time:.3f}s")
        print(f"   Compiled: {compiled_time:.3f}s")
        print(f"   Speedup: {speedup:.2f}x")
        
        return True
    except Exception as e:
        print(f"⚠️ Model compilation not available: {e}")
        return False


def test_rng_management():
    """Test RNG state management"""
    print("\n🔬 Testing RNG Management")
    print("-" * 40)
    
    try:
        # Set seed and generate
        torch.mps.manual_seed(42)
        tensor1 = torch.randn(100, device=device)
        
        # Reset seed and regenerate
        torch.mps.manual_seed(42)
        tensor2 = torch.randn(100, device=device)
        
        # Check determinism
        deterministic = torch.allclose(tensor1, tensor2)
        
        if deterministic:
            print("✅ RNG management works - deterministic")
        else:
            print("❌ RNG not deterministic")
        
        # Test state save/restore
        state = torch.mps.get_rng_state()
        random1 = torch.randn(10, device=device)
        
        torch.mps.set_rng_state(state)
        random2 = torch.randn(10, device=device)
        
        state_works = torch.allclose(random1, random2)
        
        if state_works:
            print("✅ RNG state save/restore works")
        else:
            print("❌ RNG state save/restore failed")
        
        return deterministic and state_works
        
    except Exception as e:
        print(f"❌ RNG management failed: {e}")
        return False


class SimpleMPSWorker:
    """Simple MPS worker for testing before promoting to production"""
    
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.device = device
        self.model = nn.Linear(768, 768).to(device)
        self.processed_count = 0
    
    def process(self, data: torch.Tensor) -> torch.Tensor:
        """Process data through model"""
        self.processed_count += 1
        output = self.model(data)
        return output
    
    def process_batch(self, batch: List[torch.Tensor]) -> List[torch.Tensor]:
        """Process batch without intermediate syncs"""
        results = []
        for data in batch:
            output = self.process(data)
            results.append(output)
        
        # Only sync at the end
        torch.mps.synchronize()
        return results


def test_worker_pattern():
    """Test worker pattern before promoting to actor"""
    print("\n🔬 Testing Worker Pattern")
    print("-" * 40)
    
    # Create workers
    workers = [SimpleMPSWorker(i) for i in range(4)]
    
    # Create workload
    workload = [torch.randn(768, device=device) for _ in range(40)]
    
    # Distribute work
    start = time.time()
    results = []
    for i, data in enumerate(workload):
        worker = workers[i % len(workers)]
        result = worker.process(data)
        results.append(result)
    
    torch.mps.synchronize()
    elapsed = time.time() - start
    
    # Verify
    all_processed = sum(w.processed_count for w in workers)
    
    print(f"✅ Worker pattern works")
    print(f"   Workers: {len(workers)}")
    print(f"   Items processed: {all_processed}")
    print(f"   Time: {elapsed:.3f}s")
    print(f"   Throughput: {len(workload)/elapsed:.1f} items/sec")
    
    for i, w in enumerate(workers):
        print(f"   Worker {i}: {w.processed_count} items")
    
    return all_processed == len(workload)


async def test_async_pattern():
    """Test async pattern for future actor implementation"""
    print("\n🔬 Testing Async Pattern")
    print("-" * 40)
    
    async def async_process(data: torch.Tensor, worker_id: int) -> Dict:
        """Simulate async processing"""
        model = nn.Linear(768, 768).to(device)
        
        # Process without blocking
        output = model(data)
        
        # Simulate other async work
        await asyncio.sleep(0.001)
        
        return {
            'worker_id': worker_id,
            'output_shape': output.shape,
            'output_mean': output.mean().item()
        }
    
    # Create async tasks
    workload = [torch.randn(768, device=device) for _ in range(10)]
    
    start = time.time()
    tasks = []
    for i, data in enumerate(workload):
        task = async_process(data, i % 4)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    torch.mps.synchronize()
    elapsed = time.time() - start
    
    print(f"✅ Async pattern works")
    print(f"   Tasks: {len(tasks)}")
    print(f"   Time: {elapsed:.3f}s")
    print(f"   Results: {len(results)}")
    
    return len(results) == len(workload)


def run_validation_suite():
    """Run complete validation suite"""
    print("\n" + "="*60)
    print("🚀 MPS VALIDATION SUITE")
    print("="*60)
    
    tests = [
        ("Basic Operations", test_basic_mps_operations),
        ("Parallel Execution", test_parallel_operations),
        ("Model Compilation", test_model_compilation),
        ("RNG Management", test_rng_management),
        ("Worker Pattern", test_worker_pattern),
    ]
    
    results = {}
    
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"❌ {name} failed with exception: {e}")
            results[name] = False
    
    # Run async test
    try:
        results["Async Pattern"] = asyncio.run(test_async_pattern())
    except Exception as e:
        print(f"❌ Async Pattern failed: {e}")
        results["Async Pattern"] = False
    
    # Summary
    print("\n" + "="*60)
    print("📊 VALIDATION SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name:20s}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Ready to promote to production architecture.")
    else:
        print(f"\n⚠️ {total - passed} tests failed. Fix before promoting.")
    
    return passed == total


if __name__ == "__main__":
    success = run_validation_suite()
    
    if success:
        print("\n📝 Next Steps:")
        print("  1. Create proper class hierarchy")
        print("  2. Implement base classes (Actor, Worker, Orchestrator)")
        print("  3. Add proper error handling")
        print("  4. Implement monitoring and metrics")
        print("  5. Create production configuration")