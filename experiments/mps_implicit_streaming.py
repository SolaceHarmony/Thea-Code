#!/usr/bin/env python
"""
MPS Stream-Aware Actors
Weaponizing Apple Silicon's Metal Performance Shaders for maximum parallelism

Each actor manages multiple concurrent MPS operations without explicit streams
"""

import torch
import torch.nn as nn
import ray
import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import concurrent.futures

# Force MPS
device = torch.device("mps")
print(f"🍎 Apple Silicon MPS Engaged: {device}")


@dataclass
class MPSWorkload:
    """Represents a parallel workload for MPS"""
    tensor_a: torch.Tensor
    tensor_b: torch.Tensor
    operation: str
    workload_id: int


@ray.remote(num_gpus=0.25)
class MPSStreamActor:
    """
    Actor that leverages MPS implicit streaming
    Each operation launches asynchronously on Metal
    """
    
    def __init__(self, actor_id: int):
        self.actor_id = actor_id
        self.device = torch.device("mps")
        
        # Pre-allocate tensors for different streams of work
        self.stream_buffers = {
            'tokenization': torch.zeros(32768, device=self.device),
            'embedding': torch.zeros(768, 32768, device=self.device),
            'attention': torch.zeros(12, 32768, 32768, device=self.device),
            'pattern': torch.zeros(1000, device=self.device)
        }
        
        # M2-BERT layers for parallel processing
        self.layers = nn.ModuleList([
            nn.Linear(768, 768).to(self.device) for _ in range(12)
        ])
        
        print(f"🎭 MPSStreamActor {actor_id} initialized with implicit streams")
    
    async def process_parallel_workloads(self, workloads: List[MPSWorkload]) -> Dict:
        """
        Process multiple workloads in parallel using MPS implicit streams
        No explicit stream management - Metal handles it!
        """
        
        start = time.time()
        results = []
        
        # Launch ALL operations without waiting - MPS queues them
        operations = []
        for workload in workloads:
            if workload.operation == 'matmul':
                # This returns immediately - MPS queues it
                result = torch.matmul(workload.tensor_a, workload.tensor_b)
                operations.append(('matmul', result, workload.workload_id))
                
            elif workload.operation == 'conv':
                # Convolution - also queued
                conv = nn.Conv1d(768, 768, 3, padding=1).to(self.device)
                result = conv(workload.tensor_a)
                operations.append(('conv', result, workload.workload_id))
                
            elif workload.operation == 'attention':
                # Self-attention - complex but still async
                q = self.layers[0](workload.tensor_a)
                k = self.layers[1](workload.tensor_a)
                v = self.layers[2](workload.tensor_a)
                
                scores = torch.matmul(q, k.transpose(-2, -1))
                scores = scores / torch.sqrt(torch.tensor(768.0, device=self.device))
                attn = torch.softmax(scores, dim=-1)
                result = torch.matmul(attn, v)
                operations.append(('attention', result, workload.workload_id))
        
        # Now we can do OTHER work while GPU processes
        cpu_work = self.do_cpu_work_while_gpu_runs()
        
        # Only synchronize when we need results
        torch.mps.synchronize()  # Wait for ALL MPS operations
        
        # Collect results
        for op_type, result, wl_id in operations:
            results.append({
                'workload_id': wl_id,
                'operation': op_type,
                'shape': result.shape,
                'mean': result.mean().item()
            })
        
        elapsed = time.time() - start
        
        return {
            'actor_id': self.actor_id,
            'workloads_processed': len(workloads),
            'time': elapsed,
            'throughput': len(workloads) / elapsed,
            'cpu_work': cpu_work,
            'results': results
        }
    
    def do_cpu_work_while_gpu_runs(self) -> int:
        """CPU work that happens WHILE MPS is processing"""
        count = 0
        for i in range(10000):
            count += i % 7  # Some CPU computation
        return count
    
    async def process_m2bert_parallel(self, tokens: torch.Tensor) -> Dict:
        """
        Process M2-BERT layers in parallel using MPS streams
        Each layer launches asynchronously!
        """
        
        start = time.time()
        
        # Launch ALL layers without waiting
        layer_outputs = []
        for i, layer in enumerate(self.layers):
            # This returns immediately - queued on MPS
            output = layer(tokens)
            layer_outputs.append(output)
        
        # Do pattern matching on CPU while layers process
        patterns_checked = self.check_patterns_cpu()
        
        # Only sync when needed
        torch.mps.synchronize()
        
        # Now combine results
        final = torch.stack(layer_outputs).mean(dim=0)
        
        elapsed = time.time() - start
        
        return {
            'actor_id': self.actor_id,
            'layers_processed': len(self.layers),
            'patterns_checked': patterns_checked,
            'output_shape': final.shape,
            'time': elapsed,
            'parallel_efficiency': f"{len(self.layers) / elapsed:.1f} layers/sec"
        }
    
    def check_patterns_cpu(self) -> int:
        """Pattern matching on CPU while GPU runs"""
        import re
        test_code = "if (x = = y) { return a = = b; }"
        patterns = [r'\s=\s=\s', r'>\s=', r'<\s=']
        
        count = 0
        for _ in range(1000):
            for pattern in patterns:
                matches = re.findall(pattern, test_code)
                count += len(matches)
        return count


class MPSCommunicationOrchestrator:
    """
    Orchestrates parallel communication between MPS actors
    Each actor can handle multiple streams of work concurrently
    """
    
    def __init__(self, num_actors: int = 4):
        self.actors = [MPSStreamActor.remote(i) for i in range(num_actors)]
        self.device = torch.device("mps")
        
    async def parallel_broadcast(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """
        Broadcast tensor to all actors with parallel MPS operations
        Each actor processes independently without blocking
        """
        
        # Create different workloads for each actor
        workloads = []
        for i in range(len(self.actors)):
            workload = MPSWorkload(
                tensor_a=tensor.clone(),
                tensor_b=torch.randn_like(tensor),
                operation='matmul' if i % 2 == 0 else 'conv',
                workload_id=i
            )
            workloads.append([workload] * 3)  # 3 operations per actor
        
        # Launch all actors in parallel
        tasks = []
        for actor, actor_workloads in zip(self.actors, workloads):
            task = actor.process_parallel_workloads.remote(actor_workloads)
            tasks.append(task)
        
        # Wait for all
        results = await asyncio.gather(*[asyncio.wrap_future(ray.get(t)) for t in tasks])
        
        return results
    
    async def pipeline_processing(self, data_batch: List[torch.Tensor]) -> Dict:
        """
        Pipeline processing across actors
        Actor 0 starts processing item 1 while Actor 1 processes item 0's output
        """
        
        start = time.time()
        
        # Create pipeline stages
        stage_results = []
        
        for i, data in enumerate(data_batch):
            actor = self.actors[i % len(self.actors)]
            
            # Each actor processes without waiting
            result = actor.process_m2bert_parallel.remote(data)
            stage_results.append(result)
        
        # Gather all results
        results = await asyncio.gather(*[asyncio.wrap_future(ray.get(r)) for r in stage_results])
        
        elapsed = time.time() - start
        
        return {
            'pipeline_stages': len(data_batch),
            'actors_used': len(self.actors),
            'total_time': elapsed,
            'throughput': len(data_batch) / elapsed,
            'results': results
        }


async def test_mps_streaming():
    """Test MPS implicit streaming with actors"""
    
    print("\n🔬 Testing MPS Stream-Aware Actors")
    print("=" * 60)
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    # Create orchestrator
    orchestrator = MPSCommunicationOrchestrator(num_actors=4)
    
    # Test 1: Parallel workloads
    print("\n📊 Test 1: Parallel MPS Workloads")
    test_tensor = torch.randn(768, 768, device=device)
    
    results = await orchestrator.parallel_broadcast(test_tensor)
    
    for r in results:
        print(f"  Actor {r['actor_id']}: {r['workloads_processed']} workloads in {r['time']:.3f}s")
        print(f"    Throughput: {r['throughput']:.1f} ops/sec")
        print(f"    CPU work done in parallel: {r['cpu_work']}")
    
    # Test 2: Pipeline processing
    print("\n📊 Test 2: Pipeline Processing")
    data_batch = [torch.randn(768, device=device) for _ in range(8)]
    
    pipeline_result = await orchestrator.pipeline_processing(data_batch)
    
    print(f"  Pipeline stages: {pipeline_result['pipeline_stages']}")
    print(f"  Total time: {pipeline_result['total_time']:.3f}s")
    print(f"  Throughput: {pipeline_result['throughput']:.1f} items/sec")
    
    for r in pipeline_result['results'][:3]:
        print(f"    Actor {r['actor_id']}: {r['parallel_efficiency']}")
    
    # Test 3: Stress test with many operations
    print("\n📊 Test 3: Stress Test - Many Concurrent Operations")
    
    # Create complex workloads
    complex_workloads = []
    for i in range(100):
        workload = MPSWorkload(
            tensor_a=torch.randn(768, 768, device=device),
            tensor_b=torch.randn(768, 768, device=device),
            operation='matmul' if i % 3 == 0 else 'attention',
            workload_id=i
        )
        complex_workloads.append(workload)
    
    # Process on single actor to show stream efficiency
    actor = orchestrator.actors[0]
    start = time.time()
    result = await asyncio.wrap_future(
        ray.get(actor.process_parallel_workloads.remote(complex_workloads))
    )
    elapsed = time.time() - start
    
    print(f"  Processed {len(complex_workloads)} complex operations")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Throughput: {len(complex_workloads)/elapsed:.1f} ops/sec")
    print(f"  All operations queued on MPS without explicit streams!")
    
    # Cleanup
    ray.shutdown()
    
    print("\n✅ MPS Streaming Tests Complete!")
    print("🎯 Key Insights:")
    print("  - MPS operations are implicitly streamed")
    print("  - CPU work happens WHILE GPU processes")
    print("  - torch.mps.synchronize() only when needed")
    print("  - Each actor can queue multiple operations")


async def benchmark_mps_vs_sequential():
    """Compare MPS streaming vs sequential processing"""
    
    print("\n\n⚡ MPS Streaming vs Sequential Benchmark")
    print("=" * 60)
    
    size = 2048
    num_ops = 20
    
    # Sequential processing
    print("\n📊 Sequential Processing:")
    start = time.time()
    results = []
    for i in range(num_ops):
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        c = torch.matmul(a, b)
        torch.mps.synchronize()  # Wait after EACH operation
        results.append(c.mean().item())
    seq_time = time.time() - start
    print(f"  Time: {seq_time:.3f}s")
    print(f"  Throughput: {num_ops/seq_time:.1f} ops/sec")
    
    # MPS streaming (implicit)
    print("\n📊 MPS Streaming (Implicit):")
    start = time.time()
    operations = []
    for i in range(num_ops):
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        c = torch.matmul(a, b)  # Queued, not executed yet!
        operations.append(c)
    
    # Do CPU work while GPU processes
    cpu_work = 0
    for i in range(100000):
        cpu_work += i % 13
    
    torch.mps.synchronize()  # Wait ONCE at the end
    
    results = [op.mean().item() for op in operations]
    stream_time = time.time() - start
    
    print(f"  Time: {stream_time:.3f}s")
    print(f"  Throughput: {num_ops/stream_time:.1f} ops/sec")
    print(f"  CPU work done in parallel: {cpu_work}")
    print(f"  Speedup: {seq_time/stream_time:.2f}x")
    
    print("\n🎯 MPS Streaming Advantage:")
    print(f"  - {(seq_time-stream_time)/seq_time*100:.1f}% faster")
    print(f"  - CPU remains productive during GPU ops")
    print(f"  - No explicit stream management needed!")


async def main():
    """Run all MPS tests"""
    await test_mps_streaming()
    await benchmark_mps_vs_sequential()


if __name__ == "__main__":
    asyncio.run(main())