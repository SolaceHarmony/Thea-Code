#!/usr/bin/env python
"""
Test PyTorch Scalars for ALL Math
Demonstrates that even 1+1 uses PyTorch
"""

import torch
import time

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"🖥️ Using device: {device}")


def scalar(value):
    """Convert to PyTorch scalar"""
    return torch.tensor(value, dtype=torch.float32, device=device)


def main():
    print("\n🔬 PYTORCH SCALARS FOR EVERYTHING")
    print("=" * 50)
    
    # Even 1+1 uses PyTorch!
    print("\n📐 Basic Math with PyTorch:")
    one = scalar(1)
    two = scalar(2)
    three = torch.add(one, two)
    print(f"  1 + 2 = {three.item():.0f} (computed on {device})")
    
    # More operations
    product = torch.mul(three, scalar(4))
    print(f"  3 × 4 = {product.item():.0f}")
    
    quotient = torch.div(product, scalar(2))
    print(f"  12 ÷ 2 = {quotient.item():.0f}")
    
    power = torch.pow(scalar(2), scalar(8))
    print(f"  2^8 = {power.item():.0f}")
    
    # Loop counter with PyTorch
    print("\n🔄 Loop Counter with PyTorch:")
    counter = scalar(0)
    for i in range(10):
        counter = torch.add(counter, scalar(1))
        if i % 3 == 0:
            print(f"  Iteration {i}: counter = {counter.item():.0f}")
    
    # Fibonacci with PyTorch scalars
    print("\n🔢 Fibonacci Sequence (PyTorch):")
    fib_a = scalar(0)
    fib_b = scalar(1)
    
    fib_sequence = []
    for i in range(15):
        fib_c = torch.add(fib_a, fib_b)
        fib_sequence.append(fib_c.item())
        fib_a = fib_b
        fib_b = fib_c
    
    print(f"  First 15 Fibonacci numbers:")
    print(f"  {fib_sequence[:8]}")
    print(f"  {fib_sequence[8:]}")
    
    # Statistical operations
    print("\n📊 Statistics with PyTorch:")
    numbers = [scalar(i * 0.1) for i in range(100)]
    stacked = torch.stack(numbers)
    
    mean_val = torch.mean(stacked)
    sum_val = torch.sum(stacked)
    min_val = torch.min(stacked)
    max_val = torch.max(stacked)
    std_val = torch.std(stacked)
    
    print(f"  Mean: {mean_val.item():.2f}")
    print(f"  Sum: {sum_val.item():.2f}")
    print(f"  Min: {min_val.item():.2f}")
    print(f"  Max: {max_val.item():.2f}")
    print(f"  Std: {std_val.item():.2f}")
    
    # Comparisons with PyTorch
    print("\n🔍 Comparisons with PyTorch:")
    a = scalar(42)
    b = scalar(37)
    
    print(f"  42 > 37: {torch.gt(a, b).item()}")
    print(f"  42 < 37: {torch.lt(a, b).item()}")
    print(f"  42 == 42: {torch.eq(a, scalar(42)).item()}")
    print(f"  42 != 37: {torch.ne(a, b).item()}")
    
    # Trigonometry with PyTorch
    print("\n🌊 Trigonometry with PyTorch:")
    pi = scalar(3.14159265359)
    
    sin_pi = torch.sin(pi)
    cos_pi = torch.cos(pi)
    tan_pi_4 = torch.tan(torch.div(pi, scalar(4)))
    
    print(f"  sin(π) ≈ {sin_pi.item():.6f}")
    print(f"  cos(π) ≈ {cos_pi.item():.6f}")
    print(f"  tan(π/4) ≈ {tan_pi_4.item():.6f}")
    
    # Performance comparison
    print("\n⚡ Performance Comparison (100k additions):")
    
    # Python native
    start = time.time()
    python_sum = 0
    for i in range(100000):
        python_sum = python_sum + i
    python_time = time.time() - start
    
    # PyTorch scalars
    start = time.time()
    torch_sum = scalar(0)
    for i in range(100000):
        torch_sum = torch.add(torch_sum, scalar(i))
    torch_time = time.time() - start
    
    print(f"  Python: {python_sum} in {python_time:.4f}s")
    print(f"  PyTorch: {torch_sum.item():.0f} in {torch_time:.4f}s")
    
    # Batch operations (much faster)
    print("\n🚀 Batch Operations with PyTorch:")
    start = time.time()
    numbers = torch.arange(100000, dtype=torch.float32, device=device)
    batch_sum = torch.sum(numbers)
    batch_time = time.time() - start
    
    print(f"  Batch sum: {batch_sum.item():.0f} in {batch_time:.6f}s")
    print(f"  Speedup: {torch_time/batch_time:.1f}x faster!")
    
    # Memory info
    if torch.cuda.is_available():
        print(f"\n💾 GPU Memory:")
        print(f"  Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        print(f"  Reserved: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
    
    print("\n✅ All operations performed with PyTorch scalars!")
    print(f"   Device: {device}")
    print(f"   PyTorch version: {torch.__version__}")


if __name__ == "__main__":
    main()