#!/usr/bin/env python
"""
Test Ray setup with proper configuration for our 4TB volume
Learn how to use Ray properly for our actor-first architecture
"""

import ray
import os
import json
import asyncio
import torch
from rich.console import Console

console = Console()

def setup_ray_with_large_storage():
    """Configure Ray to use the 4TB volume properly"""
    
    # Clean shutdown any existing Ray
    if ray.is_initialized():
        console.print("[yellow]Shutting down existing Ray instance...[/yellow]")
        ray.shutdown()
    
    # Create Ray temp directory on the large volume
    ray_temp_root = "/Volumes/emberstuff/ray_temp"
    os.makedirs(ray_temp_root, exist_ok=True)
    
    console.print(f"[cyan]Setting up Ray with temp directory: {ray_temp_root}[/cyan]")
    
    # Initialize Ray with simple, working configuration
    # We'll use environment variables for spilling config as a fallback
    os.environ["RAY_object_spilling_directory"] = os.path.join(ray_temp_root, "spill")
    
    ray.init(
        # Basic settings
        num_cpus=4,  # Use 4 CPUs for testing
        
        # Storage configuration
        _temp_dir=ray_temp_root,
        object_store_memory=1_000_000_000,  # 1GB object store
        
        # Logging
        logging_level="info",
        log_to_driver=True,
    )
    
    console.print("[green]✅ Ray initialized successfully![/green]")
    return ray.cluster_resources()


@ray.remote
class SimpleActor:
    """A simple Ray actor to test our setup"""
    
    def __init__(self, name):
        self.name = name
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
    def process(self, x):
        """Process using PyTorch"""
        tensor = torch.tensor(x, device=self.device)
        return (tensor * 2).cpu().item()
    
    def get_info(self):
        return {
            "name": self.name,
            "device": str(self.device),
            "pid": os.getpid()
        }


@ray.remote
class AsyncActor:
    """An async Ray actor for our architecture"""
    
    async def initialize(self):
        self.initialized = True
        return "Actor initialized"
    
    async def process_batch(self, batch):
        # Simulate async processing
        await asyncio.sleep(0.1)
        return [x * 2 for x in batch]


def test_basic_actors():
    """Test basic Ray actor functionality"""
    console.print("\n[cyan]Testing Basic Actors...[/cyan]")
    
    # Create actors
    actors = [SimpleActor.remote(f"actor_{i}") for i in range(3)]
    
    # Test processing
    futures = [actor.process.remote(i) for i, actor in enumerate(actors)]
    results = ray.get(futures)
    
    console.print(f"Results: {results}")
    assert results == [0, 2, 4], "Actor processing failed"
    
    # Get actor info
    info_futures = [actor.get_info.remote() for actor in actors]
    infos = ray.get(info_futures)
    
    for info in infos:
        console.print(f"Actor {info['name']}: PID={info['pid']}, Device={info['device']}")
    
    console.print("[green]✅ Basic actors work![/green]")


async def test_async_actors():
    """Test async Ray actors (critical for our architecture)"""
    console.print("\n[cyan]Testing Async Actors...[/cyan]")
    
    # Create async actor
    actor = AsyncActor.remote()
    
    # Initialize
    result = await actor.initialize.remote()
    console.print(f"Init result: {result}")
    
    # Process batch
    batch = [1, 2, 3, 4, 5]
    result = await actor.process_batch.remote(batch)
    console.print(f"Batch result: {result}")
    
    assert result == [2, 4, 6, 8, 10], "Async processing failed"
    
    console.print("[green]✅ Async actors work![/green]")


def test_actor_pool():
    """Test actor pool pattern (essential for distributed training)"""
    console.print("\n[cyan]Testing Actor Pool Pattern...[/cyan]")
    
    from ray.util.actor_pool import ActorPool
    
    # Create pool of actors
    actors = [SimpleActor.remote(f"pool_actor_{i}") for i in range(4)]
    pool = ActorPool(actors)
    
    # Submit work to pool
    for i in range(10):
        pool.submit(lambda actor, x: actor.process.remote(x), i)
    
    # Get results
    results = []
    while pool.has_next():
        results.append(pool.get_next())
    
    console.print(f"Pool results: {results}")
    assert len(results) == 10, "Pool didn't process all items"
    
    console.print("[green]✅ Actor pool works![/green]")


def test_large_tensor_sharing():
    """Test sharing large tensors between actors (for model weights)"""
    console.print("\n[cyan]Testing Large Tensor Sharing...[/cyan]")
    
    # Create a large tensor (simulating model weights)
    large_tensor = torch.randn(1000, 1000)  # ~4MB
    
    # Put in object store
    tensor_ref = ray.put(large_tensor)
    
    @ray.remote
    def process_shared_tensor(tensor_ref):
        # tensor_ref is already a reference, no need to ray.get it
        tensor = tensor_ref
        return tensor.mean().item()
    
    # Process with multiple actors
    futures = [process_shared_tensor.remote(tensor_ref) for _ in range(3)]
    results = ray.get(futures)
    
    console.print(f"Shared tensor means: {results}")
    
    # All should have same mean
    assert all(abs(r - results[0]) < 1e-6 for r in results), "Tensor sharing failed"
    
    console.print("[green]✅ Tensor sharing works![/green]")


def main():
    """Run all Ray tests"""
    console.print("[bold cyan]🎭 Ray Configuration Test Suite[/bold cyan]")
    console.print("Learning to use Ray properly for our actor-first architecture\n")
    
    # Setup Ray with proper configuration
    resources = setup_ray_with_large_storage()
    console.print(f"\n[cyan]Cluster resources:[/cyan] {resources}")
    
    # Run tests
    test_basic_actors()
    asyncio.run(test_async_actors())
    test_actor_pool()
    test_large_tensor_sharing()
    
    # Show Ray dashboard info
    console.print(f"\n[cyan]Ray dashboard:[/cyan] http://127.0.0.1:8265")
    console.print("[yellow]Check dashboard for memory usage and actor status[/yellow]")
    
    # Cleanup
    console.print("\n[cyan]Shutting down Ray...[/cyan]")
    ray.shutdown()
    
    console.print("\n[bold green]✨ All Ray tests passed![/bold green]")
    console.print("We now understand how to use Ray properly for our architecture!")


if __name__ == "__main__":
    main()