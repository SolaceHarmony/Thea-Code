#!/usr/bin/env python
"""
Test Actor-First Architecture
Demonstrates PyTorch scalars for ALL operations
"""

import asyncio
import sys
sys.path.insert(0, '.')

from thea_code_system.wrappers import (
    scalar, add, mul, div,
    TorchMath, TorchCounter, TorchAccumulator
)


async def test_torch_everywhere():
    """
    Test that even trivial math uses PyTorch scalars
    """
    
    print("🔬 Testing PyTorch-Everywhere Philosophy")
    print("=" * 50)
    
    # Even 1+1 uses PyTorch!
    one = scalar(1)
    two = scalar(2) 
    three = add(one, two)
    print(f"1 + 2 = {three.item()} (computed on {three.device})")
    
    # Loop counter using PyTorch
    counter = TorchCounter()
    print("\n📊 Counting to 10 with PyTorch:")
    for i in range(10):
        counter.increment()
        if i % 3 == 0:
            print(f"  Count: {counter.get().item():.0f}")
    
    # Calculate fibonacci using PyTorch scalars
    print("\n🔢 Fibonacci with PyTorch scalars:")
    fib_a = scalar(0)
    fib_b = scalar(1)
    
    for i in range(10):
        fib_c = add(fib_a, fib_b)
        if i % 3 == 0:
            print(f"  Fib[{i+2}] = {fib_c.item():.0f}")
        fib_a = fib_b
        fib_b = fib_c
    
    # Statistics with PyTorch accumulator
    print("\n📈 Statistics with PyTorch:")
    acc = TorchAccumulator()
    
    for i in range(100):
        value = mul(scalar(i), scalar(0.1))
        acc.add(value)
    
    mean = acc.mean()
    total = acc.total()
    count = acc.size()
    
    print(f"  Mean: {mean.item():.2f}")
    print(f"  Total: {total.item():.2f}")
    print(f"  Count: {count.item():.0f}")
    
    # Comparisons use PyTorch
    print("\n🔍 Comparisons with PyTorch:")
    a = scalar(42)
    b = scalar(37)
    
    is_greater = TorchMath.gt(a, b)
    is_equal = TorchMath.eq(a, scalar(42))
    
    print(f"  42 > 37: {is_greater.item()}")
    print(f"  42 == 42: {is_equal.item()}")
    
    # Mathematical operations
    print("\n🧮 Math operations with PyTorch:")
    
    x = scalar(2)
    squared = TorchMath.pow(x, scalar(2))
    sqrt_val = TorchMath.sqrt(scalar(16))
    sin_val = TorchMath.sin(TorchMath.pi())
    
    print(f"  2^2 = {squared.item():.0f}")
    print(f"  √16 = {sqrt_val.item():.0f}")
    print(f"  sin(π) ≈ {sin_val.item():.6f}")
    
    # Device info
    print(f"\n🖥️ All operations performed on: {TorchMath.device}")
    
    # Performance comparison
    import time
    
    print("\n⚡ Performance comparison:")
    
    # Python native
    start = time.time()
    python_sum = 0
    for i in range(100000):
        python_sum += i
    python_time = time.time() - start
    
    # PyTorch scalars
    start = time.time()
    torch_sum = scalar(0)
    for i in range(100000):
        torch_sum = add(torch_sum, scalar(i))
    torch_time = time.time() - start
    
    print(f"  Python sum: {python_sum} in {python_time:.4f}s")
    print(f"  Torch sum: {torch_sum.item():.0f} in {torch_time:.4f}s")
    
    # Show acceleration on GPU if available
    import torch
    if torch.cuda.is_available():
        print(f"  🚀 GPU acceleration available!")
    elif torch.backends.mps.is_available():
        print(f"  🚀 Apple Silicon acceleration available!")
    else:
        print(f"  💻 Running on CPU")
    
    print("\n✅ All operations use PyTorch scalars - Actor-First Architecture Ready!")


async def test_actor_system():
    """
    Test the actor system with PyTorch scalars
    """
    
    print("\n\n🎭 Testing Actor System")
    print("=" * 50)
    
    from thea_code_system.wrappers import (
        Actor, async_actor, remote_method,
        create_actor, ActorPool
    )
    
    # Define a simple test actor
    @async_actor(num_cpus=1)
    class ComputeActor(Actor):
        def __init__(self, name: str = None):
            super().__init__(name)
            self.compute_count = TorchCounter()
        
        @remote_method
        async def compute(self, x: float, y: float) -> float:
            """All math uses PyTorch scalars"""
            self.compute_count.increment()
            
            x_tensor = scalar(x)
            y_tensor = scalar(y)
            
            # Complex calculation using PyTorch
            result = add(
                TorchMath.pow(x_tensor, scalar(2)),
                TorchMath.pow(y_tensor, scalar(2))
            )
            result = TorchMath.sqrt(result)
            
            return result.item()
        
        @remote_method
        async def get_stats(self):
            return {
                'name': self.name,
                'computations': self.compute_count.get().item()
            }
    
    # Create actor pool
    print("Creating actor pool with 3 workers...")
    pool = ActorPool(ComputeActor, num_actors=3)
    
    # Test computations
    print("\n🔄 Running distributed computations:")
    
    tasks = []
    for i in range(10):
        actor = pool.get_next()
        result = await actor.call('compute', i, i+1)
        print(f"  Actor computed: √({i}² + {i+1}²) = {result:.2f}")
    
    # Get stats from all actors
    print("\n📊 Actor statistics:")
    stats = await pool.broadcast('get_stats')
    for stat in stats:
        print(f"  {stat['name']}: {stat['computations']} computations")
    
    # Health check
    print("\n🏥 Pool health check:")
    health = await pool.health_check()
    print(f"  Healthy actors: {health['healthy_actors']}/{health['pool_size']}")
    
    # Cleanup
    pool.shutdown()
    print("\n✅ Actor system test complete!")


async def main():
    """Run all tests"""
    await test_torch_everywhere()
    await test_actor_system()


if __name__ == "__main__":
    asyncio.run(main())