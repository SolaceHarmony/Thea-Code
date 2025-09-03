#!/usr/bin/env python
"""
TRUE MPS Architecture with Full API
Using ALL MPS-specific features for maximum Apple Silicon performance

This is the REAL DEAL - using torch.mps API fully!
"""

import torch
import torch.mps
import torch.nn as nn
import ray
import asyncio
import time
import os
from typing import Dict, List, Optional, Any, Tuple
from contextlib import contextmanager

# Check MPS availability
if not torch.backends.mps.is_available():
    raise RuntimeError("MPS not available! This requires Apple Silicon Mac")

device = torch.device("mps")
print(f"🍎 Apple Silicon MPS - Full API Enabled")
print(f"📱 MPS Devices Available: {torch.mps.device_count()}")
print(f"💾 Current Allocated: {torch.mps.current_allocated_memory() / 1024**2:.2f} MB")
print(f"💾 Driver Allocated: {torch.mps.driver_allocated_memory() / 1024**2:.2f} MB")
print(f"💾 Recommended Max: {torch.mps.recommended_max_memory() / 1024**3:.2f} GB")

# Set memory fraction to prevent OOM
torch.mps.set_per_process_memory_fraction(0.8)

# Set deterministic seed for reproducibility
torch.mps.manual_seed(42)


@contextmanager
def mps_profile(name: str):
    """Context manager for MPS profiling with OS Signposts"""
    # MPS profiler expects specific mode strings
    mode = "interval"  # Use interval mode for profiling
    if torch.mps.profiler.is_metal_capture_enabled():
        with torch.mps.profiler.metal_capture():
            with torch.mps.profiler.profile(mode):
                yield
    else:
        with torch.mps.profiler.profile(mode):
            yield


class MPSEvent:
    """Wrapper for MPS events for fine-grained synchronization"""
    
    def __init__(self):
        self.event = torch.mps.event.Event()
    
    def record(self):
        """Record event in current stream"""
        self.event.record()
    
    def wait(self):
        """Wait for event to complete"""
        self.event.wait()
    
    def synchronize(self):
        """Synchronize on this event"""
        self.event.synchronize()
    
    def query(self) -> bool:
        """Check if event has been recorded"""
        return self.event.query()


@ray.remote(num_gpus=0.25)
class MPSTrueActor:
    """
    Actor using full MPS API capabilities
    - Memory management
    - Event synchronization
    - Profiling
    - RNG state management
    """
    
    def __init__(self, actor_id: int):
        self.actor_id = actor_id
        self.device = torch.device("mps")
        
        # Save and set RNG state for reproducibility
        self.rng_state = torch.mps.get_rng_state()
        torch.mps.manual_seed(42 + actor_id)
        
        # Track memory usage
        self.initial_memory = torch.mps.current_allocated_memory()
        
        # Create events for synchronization
        self.events = [MPSEvent() for _ in range(4)]
        
        # Models
        self.model = nn.Sequential(
            nn.Linear(768, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 768)
        ).to(self.device)
        
        # Compile model for better performance
        self.compiled_model = torch.compile(self.model, backend="aot_eager")
        
        print(f"🎭 MPSTrueActor {actor_id} initialized")
        print(f"  Memory allocated: {(torch.mps.current_allocated_memory() - self.initial_memory) / 1024**2:.2f} MB")
    
    async def process_with_events(self, tensors: List[torch.Tensor]) -> Dict:
        """
        Process tensors using MPS events for synchronization
        """
        
        with mps_profile(f"Actor_{self.actor_id}_Processing"):
            start = time.time()
            results = []
            
            # Process each tensor with event tracking
            for i, tensor in enumerate(tensors):
                event_idx = i % len(self.events)
                event = self.events[event_idx]
                
                # Process
                with mps_profile(f"Tensor_{i}"):
                    output = self.compiled_model(tensor)
                    
                    # Record completion
                    event.record()
                    
                    results.append({
                        'output': output,
                        'event': event
                    })
            
            # Wait for all events
            for result in results:
                result['event'].wait()
            
            # Synchronize MPS
            torch.mps.synchronize()
            
            elapsed = time.time() - start
            
            # Memory stats
            current_mem = torch.mps.current_allocated_memory()
            peak_mem = torch.mps.driver_allocated_memory()
            
            return {
                'actor_id': self.actor_id,
                'items_processed': len(tensors),
                'time': elapsed,
                'throughput': len(tensors) / elapsed,
                'memory_used_mb': (current_mem - self.initial_memory) / 1024**2,
                'peak_memory_mb': peak_mem / 1024**2
            }
    
    async def memory_stress_test(self, size: int, iterations: int) -> Dict:
        """
        Stress test memory allocation and caching
        """
        
        with mps_profile(f"Memory_Stress_{self.actor_id}"):
            start = time.time()
            
            # Track memory
            allocations = []
            
            for i in range(iterations):
                # Allocate large tensor
                tensor = torch.randn(size, size, device=self.device)
                result = torch.matmul(tensor, tensor)
                
                # Track allocation
                allocations.append(torch.mps.current_allocated_memory())
                
                # Periodically clear cache
                if i % 10 == 0:
                    torch.mps.empty_cache()
            
            # Final memory state
            torch.mps.synchronize()
            torch.mps.empty_cache()
            
            elapsed = time.time() - start
            
            return {
                'actor_id': self.actor_id,
                'iterations': iterations,
                'tensor_size': f"{size}x{size}",
                'time': elapsed,
                'peak_allocation_mb': max(allocations) / 1024**2,
                'final_allocation_mb': torch.mps.current_allocated_memory() / 1024**2,
                'cache_cleared': iterations // 10
            }
    
    async def parallel_kernels_test(self, num_kernels: int) -> Dict:
        """
        Launch multiple kernels without synchronization
        MPS will schedule them optimally
        """
        
        with mps_profile(f"Parallel_Kernels_{self.actor_id}"):
            start = time.time()
            
            # Launch kernels without waiting
            kernels = []
            for i in range(num_kernels):
                a = torch.randn(1024, 1024, device=self.device)
                b = torch.randn(1024, 1024, device=self.device)
                
                # These operations are queued, not executed
                c = torch.matmul(a, b)
                d = torch.relu(c)
                e = torch.softmax(d, dim=-1)
                
                kernels.append(e)
            
            # Do CPU work while GPU processes
            cpu_work = sum(i * i for i in range(10000))
            
            # Now synchronize
            torch.mps.synchronize()
            
            elapsed = time.time() - start
            
            # Verify results
            results_valid = all(k.shape == (1024, 1024) for k in kernels)
            
            return {
                'actor_id': self.actor_id,
                'kernels_launched': num_kernels,
                'time': elapsed,
                'kernels_per_second': num_kernels / elapsed,
                'cpu_work_done': cpu_work,
                'results_valid': results_valid
            }
    
    async def rng_determinism_test(self) -> Dict:
        """
        Test RNG state management for reproducibility
        """
        
        # Save current state
        saved_state = torch.mps.get_rng_state()
        
        # Generate random numbers
        torch.mps.manual_seed(123)
        random1 = torch.randn(100, device=self.device)
        
        # Reset and regenerate
        torch.mps.manual_seed(123)
        random2 = torch.randn(100, device=self.device)
        
        # Check determinism
        deterministic = torch.allclose(random1, random2)
        
        # Restore state
        torch.mps.set_rng_state(saved_state)
        
        return {
            'actor_id': self.actor_id,
            'deterministic': deterministic,
            'state_size': len(saved_state)
        }


class MPSOrchestrator:
    """
    Orchestrator using full MPS capabilities
    """
    
    def __init__(self, num_actors: int = 4):
        self.num_actors = num_actors
        self.actors = [MPSTrueActor.remote(i) for i in range(num_actors)]
        
        # Global memory management
        torch.mps.empty_cache()
        
        print(f"🎼 MPS Orchestrator initialized with {num_actors} actors")
        print(f"  Total MPS memory: {torch.mps.recommended_max_memory() / 1024**3:.2f} GB")
    
    async def distributed_processing(self, workload_size: int) -> Dict:
        """
        Distribute processing across actors with memory management
        """
        
        with mps_profile("Distributed_Processing"):
            # Clear cache before starting
            torch.mps.empty_cache()
            
            # Create workloads
            workloads = []
            for i in range(self.num_actors):
                actor_work = [
                    torch.randn(768, device=device) 
                    for _ in range(workload_size)
                ]
                workloads.append(actor_work)
            
            # Launch all actors
            start = time.time()
            
            tasks = []
            for actor, work in zip(self.actors, workloads):
                task = actor.process_with_events.remote(work)
                tasks.append(task)
            
            # Wait for completion
            results = await asyncio.gather(*[asyncio.wrap_future(ray.get(t)) for t in tasks])
            
            elapsed = time.time() - start
            
            # Aggregate metrics
            total_processed = sum(r['items_processed'] for r in results)
            total_memory = sum(r['memory_used_mb'] for r in results)
            
            # Clear cache after processing
            torch.mps.empty_cache()
            
            return {
                'actors': self.num_actors,
                'total_items': total_processed,
                'time': elapsed,
                'throughput': total_processed / elapsed,
                'total_memory_mb': total_memory,
                'results': results
            }
    
    async def memory_coordination_test(self) -> Dict:
        """
        Test coordinated memory management across actors
        """
        
        # Set conservative memory fraction
        torch.mps.set_per_process_memory_fraction(0.5)
        
        # Run memory stress test on all actors
        tasks = []
        for actor in self.actors:
            task = actor.memory_stress_test.remote(size=2048, iterations=20)
            tasks.append(task)
        
        results = await asyncio.gather(*[asyncio.wrap_future(ray.get(t)) for t in tasks])
        
        # Reset memory fraction
        torch.mps.set_per_process_memory_fraction(0.8)
        
        return {
            'actors_tested': len(results),
            'results': results
        }


async def test_true_mps_architecture():
    """Test the true MPS architecture with full API"""
    
    print("\n🔬 Testing TRUE MPS Architecture")
    print("=" * 60)
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    # Create orchestrator
    orchestrator = MPSOrchestrator(num_actors=4)
    
    # Test 1: Event-based processing
    print("\n📊 Test 1: MPS Event-Based Processing")
    result = await orchestrator.distributed_processing(workload_size=10)
    
    print(f"  Total items: {result['total_items']}")
    print(f"  Time: {result['time']:.3f}s")
    print(f"  Throughput: {result['throughput']:.1f} items/sec")
    print(f"  Memory used: {result['total_memory_mb']:.2f} MB")
    
    for r in result['results']:
        print(f"    Actor {r['actor_id']}: {r['throughput']:.1f} items/sec, {r['memory_used_mb']:.2f} MB")
    
    # Test 2: Memory coordination
    print("\n📊 Test 2: Coordinated Memory Management")
    mem_result = await orchestrator.memory_coordination_test()
    
    for r in mem_result['results']:
        print(f"  Actor {r['actor_id']}:")
        print(f"    Peak memory: {r['peak_allocation_mb']:.2f} MB")
        print(f"    Cache clears: {r['cache_cleared']}")
    
    # Test 3: Parallel kernels
    print("\n📊 Test 3: Parallel Kernel Execution")
    
    actor = orchestrator.actors[0]
    kernel_result = await asyncio.wrap_future(
        ray.get(actor.parallel_kernels_test.remote(num_kernels=50))
    )
    
    print(f"  Kernels launched: {kernel_result['kernels_launched']}")
    print(f"  Time: {kernel_result['time']:.3f}s")
    print(f"  Kernels/second: {kernel_result['kernels_per_second']:.1f}")
    print(f"  CPU work done in parallel: {kernel_result['cpu_work_done']}")
    
    # Test 4: RNG determinism
    print("\n📊 Test 4: RNG State Management")
    
    rng_result = await asyncio.wrap_future(
        ray.get(actor.rng_determinism_test.remote())
    )
    
    print(f"  Deterministic: {rng_result['deterministic']}")
    print(f"  RNG state size: {rng_result['state_size']} bytes")
    
    # Final memory status
    print("\n💾 Final Memory Status:")
    print(f"  Current allocated: {torch.mps.current_allocated_memory() / 1024**2:.2f} MB")
    print(f"  Driver allocated: {torch.mps.driver_allocated_memory() / 1024**2:.2f} MB")
    
    # Clear cache
    torch.mps.empty_cache()
    print(f"  After cache clear: {torch.mps.current_allocated_memory() / 1024**2:.2f} MB")
    
    # Cleanup
    ray.shutdown()
    
    print("\n✅ TRUE MPS Architecture Test Complete!")
    print("\n🎯 Key Features Demonstrated:")
    print("  ✓ MPS Events for synchronization")
    print("  ✓ Memory management with cache control")
    print("  ✓ RNG state management for reproducibility")
    print("  ✓ Profiling with OS Signposts")
    print("  ✓ Parallel kernel execution")
    print("  ✓ Memory fraction control")


async def benchmark_mps_features():
    """Benchmark MPS-specific features"""
    
    print("\n\n⚡ MPS Feature Benchmarks")
    print("=" * 60)
    
    # Benchmark 1: Synchronization methods
    print("\n📊 Synchronization Methods:")
    
    size = 2048
    iterations = 100
    
    # No sync
    start = time.time()
    for _ in range(iterations):
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        c = torch.matmul(a, b)
    # No sync here!
    no_sync_time = time.time() - start
    
    # With sync
    start = time.time()
    for _ in range(iterations):
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        c = torch.matmul(a, b)
        torch.mps.synchronize()
    sync_time = time.time() - start
    
    print(f"  No sync: {no_sync_time:.3f}s ({iterations/no_sync_time:.1f} ops/sec)")
    print(f"  With sync: {sync_time:.3f}s ({iterations/sync_time:.1f} ops/sec)")
    print(f"  Speedup without sync: {sync_time/no_sync_time:.2f}x")
    
    # Benchmark 2: Memory cache impact
    print("\n📊 Memory Cache Impact:")
    
    # With cache clearing
    start = time.time()
    for i in range(50):
        tensor = torch.randn(4096, 4096, device=device)
        result = tensor @ tensor.T
        if i % 10 == 0:
            torch.mps.empty_cache()
    cache_clear_time = time.time() - start
    
    # Without cache clearing
    start = time.time()
    for i in range(50):
        tensor = torch.randn(4096, 4096, device=device)
        result = tensor @ tensor.T
    no_cache_clear_time = time.time() - start
    
    print(f"  With cache clearing: {cache_clear_time:.3f}s")
    print(f"  Without clearing: {no_cache_clear_time:.3f}s")
    print(f"  Overhead: {(cache_clear_time-no_cache_clear_time)/no_cache_clear_time*100:.1f}%")
    
    print("\n✅ MPS Benchmarks Complete!")


async def main():
    """Run all MPS tests"""
    await test_true_mps_architecture()
    await benchmark_mps_features()


if __name__ == "__main__":
    asyncio.run(main())