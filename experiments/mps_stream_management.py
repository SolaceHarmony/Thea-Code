#!/usr/bin/env python
"""
MPS Explicit Streams with Actors
Using torch.Stream for fine-grained control over MPS parallelism

This is the REAL architecture - explicit streams per actor!
"""

import torch
import torch.nn as nn
import ray
import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import os

# Enable MPS fallback for unsupported ops
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Force MPS
if not torch.backends.mps.is_available():
    print("⚠️ MPS not available, using CPU")
    device = torch.device("cpu")
else:
    device = torch.device("mps")
    print(f"🍎 Apple Silicon MPS with Explicit Streams: {device}")


@ray.remote(num_gpus=0.25)
class MPSMultiStreamActor:
    """
    Actor with multiple explicit MPS streams
    Each stream handles different types of workloads concurrently
    """
    
    def __init__(self, actor_id: int, num_streams: int = 4):
        self.actor_id = actor_id
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.num_streams = num_streams
        
        # Create multiple explicit streams
        self.streams = []
        for i in range(num_streams):
            stream = torch.Stream(device=self.device)
            self.streams.append(stream)
            print(f"  Stream {i} created for Actor {actor_id}")
        
        # Different models for different streams
        self.models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(768, 1024),
                nn.ReLU(),
                nn.Linear(1024, 768)
            ).to(self.device) for _ in range(num_streams)
        ])
        
        # Events for synchronization
        self.events = [stream.record_event() for stream in self.streams]
        
        print(f"🎭 MPSMultiStreamActor {actor_id} initialized with {num_streams} explicit streams")
    
    async def parallel_stream_processing(self, workloads: List[torch.Tensor]) -> Dict:
        """
        Process workloads across multiple explicit streams
        Each stream runs independently and concurrently
        """
        
        start = time.time()
        results = []
        stream_times = []
        
        # Distribute workloads across streams
        for i, workload in enumerate(workloads):
            stream_idx = i % self.num_streams
            stream = self.streams[stream_idx]
            model = self.models[stream_idx]
            
            # Use the stream context
            with torch.Stream(device=self.device) as stream:
                stream_start = time.time()
                
                # Operations within this stream
                output = model(workload)
                
                # Additional operations in same stream
                normalized = torch.nn.functional.normalize(output, dim=-1)
                activated = torch.relu(normalized)
                
                # Record event for this stream
                event = stream.record_event()
                
                # Store result (tensor stays on device)
                results.append({
                    'stream_id': stream_idx,
                    'output': activated,
                    'event': event
                })
                
                stream_times.append(time.time() - stream_start)
        
        # Wait for all streams to complete
        for result in results:
            result['event'].wait()
        
        # Synchronize all streams
        for stream in self.streams:
            stream.synchronize()
        
        elapsed = time.time() - start
        
        # Compute metrics with torch scalars
        outputs = torch.stack([r['output'] for r in results])
        mean_output = outputs.mean()
        
        return {
            'actor_id': self.actor_id,
            'num_streams': self.num_streams,
            'workloads_processed': len(workloads),
            'total_time': elapsed,
            'throughput': len(workloads) / elapsed,
            'mean_output': mean_output.item(),
            'stream_times': stream_times,
            'parallel_efficiency': len(workloads) / (elapsed * self.num_streams)
        }
    
    async def stream_communication_test(self, tensor: torch.Tensor) -> Dict:
        """
        Test inter-stream communication using events
        Stream 0 -> Stream 1 -> Stream 2 -> Stream 3
        """
        
        start = time.time()
        intermediate_results = []
        
        # Pipeline through streams
        current_tensor = tensor
        
        for i, (stream, model) in enumerate(zip(self.streams, self.models)):
            with torch.Stream(device=self.device) as stream:
                # Process in this stream
                output = model(current_tensor)
                
                # Record completion event
                event = stream.record_event()
                
                # Next stream waits for this event
                if i < len(self.streams) - 1:
                    self.streams[i + 1].wait_event(event)
                
                current_tensor = output
                intermediate_results.append(output.mean().item())
        
        # Final synchronization
        self.streams[-1].synchronize()
        
        elapsed = time.time() - start
        
        return {
            'actor_id': self.actor_id,
            'pipeline_stages': len(self.streams),
            'time': elapsed,
            'intermediate_results': intermediate_results,
            'final_output_mean': current_tensor.mean().item()
        }
    
    async def concurrent_pattern_matching(self, code_batch: List[str]) -> Dict:
        """
        Use different streams for different pattern types
        Stream 0: Spaced equals
        Stream 1: Arrow functions
        Stream 2: Operators
        Stream 3: Brackets
        """
        
        import re
        
        patterns = [
            (r'\s=\s=\s', 'spaced-equals'),
            (r'\s=>\s', 'arrow-functions'),
            (r'[><=]\s=', 'operators'),
            (r'\[\s\]|\(\s\)', 'brackets')
        ]
        
        start = time.time()
        stream_results = []
        
        # Each stream handles different pattern type
        for stream_idx, (pattern, name) in enumerate(patterns):
            stream = self.streams[stream_idx]
            
            with torch.Stream(device=self.device) as stream:
                matches_per_file = []
                
                for code in code_batch:
                    matches = len(re.findall(pattern, code))
                    matches_per_file.append(matches)
                
                # Convert to tensor for GPU processing
                match_tensor = torch.tensor(matches_per_file, 
                                           dtype=torch.float32, 
                                           device=self.device)
                
                # Statistical analysis on GPU
                mean_matches = match_tensor.mean()
                std_matches = match_tensor.std()
                max_matches = match_tensor.max()
                
                # Record completion
                event = stream.record_event()
                
                stream_results.append({
                    'pattern': name,
                    'stream_id': stream_idx,
                    'total_matches': match_tensor.sum().item(),
                    'mean': mean_matches.item(),
                    'std': std_matches.item(),
                    'max': max_matches.item(),
                    'event': event
                })
        
        # Wait for all pattern streams
        for result in stream_results:
            result['event'].wait()
            del result['event']  # Remove event from results
        
        elapsed = time.time() - start
        
        return {
            'actor_id': self.actor_id,
            'files_processed': len(code_batch),
            'patterns_checked': len(patterns),
            'time': elapsed,
            'files_per_second': len(code_batch) / elapsed,
            'stream_results': stream_results
        }


class MPSStreamOrchestrator:
    """
    Orchestrates actors with explicit MPS streams
    Creates complex parallel workflows
    """
    
    def __init__(self, num_actors: int = 4, streams_per_actor: int = 4):
        self.num_actors = num_actors
        self.streams_per_actor = streams_per_actor
        self.actors = [
            MPSMultiStreamActor.remote(i, streams_per_actor) 
            for i in range(num_actors)
        ]
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        
        print(f"🎼 Orchestrator initialized:")
        print(f"  Actors: {num_actors}")
        print(f"  Streams per actor: {streams_per_actor}")
        print(f"  Total parallel streams: {num_actors * streams_per_actor}")
    
    async def distributed_stream_processing(self, data_batch: List[torch.Tensor]) -> Dict:
        """
        Distribute data across actors, each using multiple streams
        Total parallelism = num_actors * streams_per_actor
        """
        
        start = time.time()
        
        # Distribute data to actors
        actor_workloads = [[] for _ in range(self.num_actors)]
        for i, data in enumerate(data_batch):
            actor_idx = i % self.num_actors
            actor_workloads[actor_idx].append(data)
        
        # Launch parallel processing on all actors
        tasks = []
        for actor, workloads in zip(self.actors, actor_workloads):
            if workloads:  # Only if actor has work
                task = actor.parallel_stream_processing.remote(workloads)
                tasks.append(task)
        
        # Wait for all actors
        results = await asyncio.gather(*[asyncio.wrap_future(ray.get(t)) for t in tasks])
        
        elapsed = time.time() - start
        
        # Aggregate metrics
        total_processed = sum(r['workloads_processed'] for r in results)
        avg_throughput = sum(r['throughput'] for r in results) / len(results)
        
        return {
            'total_items': len(data_batch),
            'actors_used': len(results),
            'total_streams': self.num_actors * self.streams_per_actor,
            'total_time': elapsed,
            'items_per_second': total_processed / elapsed,
            'avg_throughput_per_actor': avg_throughput,
            'results': results
        }


async def test_explicit_streams():
    """Test explicit MPS streams"""
    
    print("\n🔬 Testing Explicit MPS Streams")
    print("=" * 60)
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    # Test 1: Single actor with multiple streams
    print("\n📊 Test 1: Single Actor, Multiple Streams")
    actor = MPSMultiStreamActor.remote(0, num_streams=4)
    
    # Create workloads
    workloads = [torch.randn(768, device=device) for _ in range(16)]
    
    result = await asyncio.wrap_future(
        ray.get(actor.parallel_stream_processing.remote(workloads))
    )
    
    print(f"  Actor 0 with {result['num_streams']} streams:")
    print(f"    Processed: {result['workloads_processed']} items")
    print(f"    Time: {result['total_time']:.3f}s")
    print(f"    Throughput: {result['throughput']:.1f} items/sec")
    print(f"    Parallel efficiency: {result['parallel_efficiency']:.2f}")
    
    # Test 2: Stream communication
    print("\n📊 Test 2: Inter-Stream Communication Pipeline")
    test_tensor = torch.randn(768, device=device)
    
    comm_result = await asyncio.wrap_future(
        ray.get(actor.stream_communication_test.remote(test_tensor))
    )
    
    print(f"  Pipeline with {comm_result['pipeline_stages']} stages:")
    print(f"    Time: {comm_result['time']:.3f}s")
    print(f"    Stage outputs: {comm_result['intermediate_results']}")
    
    # Test 3: Orchestrated multi-actor, multi-stream
    print("\n📊 Test 3: Multi-Actor, Multi-Stream Orchestration")
    orchestrator = MPSStreamOrchestrator(num_actors=4, streams_per_actor=4)
    
    # Large batch of data
    large_batch = [torch.randn(768, device=device) for _ in range(64)]
    
    orch_result = await orchestrator.distributed_stream_processing(large_batch)
    
    print(f"  Total parallelism: {orch_result['total_streams']} streams")
    print(f"  Items processed: {orch_result['total_items']}")
    print(f"  Time: {orch_result['total_time']:.3f}s")
    print(f"  Throughput: {orch_result['items_per_second']:.1f} items/sec")
    
    # Test 4: Pattern matching with streams
    print("\n📊 Test 4: Concurrent Pattern Matching")
    
    # Generate test code
    test_code = [
        "if (x = = y) { return a = = b; }" for _ in range(100)
    ]
    
    pattern_result = await asyncio.wrap_future(
        ray.get(actor.concurrent_pattern_matching.remote(test_code))
    )
    
    print(f"  Files processed: {pattern_result['files_processed']}")
    print(f"  Patterns checked: {pattern_result['patterns_checked']}")
    print(f"  Time: {pattern_result['time']:.3f}s")
    print(f"  Files/second: {pattern_result['files_per_second']:.1f}")
    
    for sr in pattern_result['stream_results']:
        print(f"    Stream {sr['stream_id']} ({sr['pattern']}): {sr['total_matches']} matches")
    
    # Cleanup
    ray.shutdown()
    
    print("\n✅ Explicit MPS Streams Test Complete!")
    print("\n🎯 Architecture Insights:")
    print("  - Each actor has multiple explicit streams")
    print("  - Streams can communicate via events")
    print("  - Total parallelism = actors × streams")
    print("  - Pattern matching parallelized across streams")


async def main():
    """Run MPS stream tests"""
    await test_explicit_streams()


if __name__ == "__main__":
    asyncio.run(main())