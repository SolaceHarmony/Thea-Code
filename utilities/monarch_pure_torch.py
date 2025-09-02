#!/usr/bin/env python3
"""
Pure PyTorch Monarch Matrix Implementation
No numpy, full GPU support, proper autograd

This is the real deal - butterfly transforms, block diagonal matrices,
all with proper PyTorch tensor operations and Ray actor integration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math
import ray
from dataclasses import dataclass
import time

@dataclass 
class MonarchConfig:
    """Configuration for Monarch matrices"""
    size: int
    n_blocks: int = 16
    dtype: torch.dtype = torch.float32
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

class ButterflyLayer(nn.Module):
    """
    Single butterfly layer for O(log n) layers total
    Pure PyTorch implementation
    """
    
    def __init__(self, size: int, stride: int, device: str = "cuda"):
        super().__init__()
        self.size = size
        self.stride = stride
        self.device = device
        
        # Number of 2x2 butterfly units at this layer
        n_butterflies = size // 2
        
        # Initialize butterfly matrices (2x2 blocks)
        # Using real-valued parameterization
        self.W_real = nn.Parameter(torch.randn(n_butterflies, 2, 2, device=device) / math.sqrt(2))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply butterfly layer
        x: [batch_size, size]
        """
        batch_size = x.shape[0]
        
        # Reshape to expose butterfly structure
        # We process pairs of elements
        x_reshaped = x.view(batch_size, -1, 2)  # [batch, size/2, 2]
        
        # Apply butterfly transformations
        # Each 2x2 matrix operates on a pair
        x_transformed = torch.einsum('bni,nij->bnj', x_reshaped, self.W_real)
        
        # Reshape back
        return x_transformed.view(batch_size, -1)

class ButterflyTransform(nn.Module):
    """
    Complete butterfly transform achieving O(n log n) complexity
    Pure PyTorch, GPU-friendly
    """
    
    def __init__(self, size: int, device: str = "cuda"):
        super().__init__()
        
        # Size must be power of 2 for butterfly
        assert size > 0 and (size & (size - 1)) == 0, "Size must be power of 2"
        
        self.size = size
        self.device = device
        self.log_n = int(torch.log2(torch.tensor(size, dtype=torch.float32)).item())
        
        # Create log(n) butterfly layers
        self.layers = nn.ModuleList()
        for i in range(self.log_n):
            stride = 2 ** (self.log_n - i - 1)
            self.layers.append(ButterflyLayer(size, stride, device))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply complete butterfly transform
        Complexity: O(n log n)
        """
        for layer in self.layers:
            x = layer(x)
        return x

class BlockDiagonal(nn.Module):
    """
    Block diagonal matrix for the core of Monarch decomposition
    Pure PyTorch implementation
    """
    
    def __init__(self, size: int, n_blocks: int, device: str = "cuda"):
        super().__init__()
        
        assert size % n_blocks == 0, f"Size {size} must be divisible by n_blocks {n_blocks}"
        
        self.size = size
        self.n_blocks = n_blocks
        self.block_size = size // n_blocks
        self.device = device
        
        # Initialize blocks
        self.blocks = nn.Parameter(
            torch.randn(n_blocks, self.block_size, self.block_size, device=device) 
            / math.sqrt(self.block_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply block diagonal matrix
        Complexity: O(n * block_size) = O(n^(3/2)) when block_size = sqrt(n)
        """
        batch_size = x.shape[0]
        
        # Reshape to blocks
        x_blocks = x.view(batch_size, self.n_blocks, self.block_size)
        
        # Apply block diagonal multiplication
        # Using einsum for efficiency
        y_blocks = torch.einsum('bni,nij->bnj', x_blocks, self.blocks)
        
        # Reshape back
        return y_blocks.view(batch_size, self.size)

class MonarchMatrix(nn.Module):
    """
    Complete Monarch Matrix: M = P @ B @ Q
    
    P, Q: Butterfly transforms (O(n log n))
    B: Block diagonal (O(n^(3/2)))
    
    Total complexity: O(n log n) or O(n^(3/2)) depending on size
    """
    
    def __init__(self, config: MonarchConfig):
        super().__init__()
        
        self.size = config.size
        self.n_blocks = config.n_blocks
        self.device = config.device
        
        # Check if we can use butterfly (power of 2)
        self.use_butterfly = (self.size & (self.size - 1)) == 0
        
        if self.use_butterfly:
            # Use butterfly transforms for P and Q
            self.P = ButterflyTransform(self.size, self.device)
            self.Q = ButterflyTransform(self.size, self.device)
        else:
            # Fall back to learned linear (still better than dense)
            self.P = nn.Linear(self.size, self.size, bias=False, device=self.device)
            self.Q = nn.Linear(self.size, self.size, bias=False, device=self.device)
            
            # Initialize as near-orthogonal
            nn.init.orthogonal_(self.P.weight)
            nn.init.orthogonal_(self.Q.weight)
        
        # Block diagonal core
        self.B = BlockDiagonal(self.size, self.n_blocks, self.device)
        
        # Move to device
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Monarch matrix multiplication
        
        Input: [batch_size, size] or [batch_size, seq_len, size]
        Output: Same shape as input
        """
        # Handle both 2D and 3D inputs
        original_shape = x.shape
        if x.dim() == 3:
            batch_size, seq_len, dim = x.shape
            x = x.reshape(-1, dim)
        
        # Ensure on correct device
        x = x.to(self.device)
        
        # Apply P
        x = self.P(x)
        
        # Apply B (block diagonal)
        x = self.B(x)
        
        # Apply Q
        x = self.Q(x)
        
        # Restore original shape
        if len(original_shape) == 3:
            x = x.reshape(original_shape)
        
        return x
    
    def compute_flops(self) -> int:
        """Compute FLOPs for one forward pass"""
        n = self.size
        
        if self.use_butterfly:
            # Butterfly: O(n log n) for P and Q
            log_n = int(torch.log2(torch.tensor(n, dtype=torch.float32)).item())
            butterfly_flops = 2 * n * log_n * 6  # 6 ops per butterfly unit
        else:
            # Linear: O(n^2) for P and Q
            butterfly_flops = 2 * n * n * 2
        
        # Block diagonal: O(n * block_size)
        block_size = n // self.n_blocks
        block_flops = n * block_size * 2
        
        return butterfly_flops + block_flops

@ray.remote(num_gpus=0.5 if torch.cuda.is_available() else 0)
class MonarchActor:
    """
    Ray actor for distributed Monarch matrix operations
    Pure PyTorch, GPU-enabled
    """
    
    def __init__(self, config: MonarchConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create Monarch matrix
        self.monarch = MonarchMatrix(config)
        self.monarch = self.monarch.to(self.device)
        
        # Optimizer for distributed training
        self.optimizer = torch.optim.Adam(self.monarch.parameters(), lr=1e-3)
        
        print(f"MonarchActor initialized on {self.device}")
        print(f"Parameters: {sum(p.numel() for p in self.monarch.parameters()):,}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = x.to(self.device)
        with torch.no_grad():
            output = self.monarch(x)
        return output.cpu()
    
    def train_step(self, x: torch.Tensor, target: torch.Tensor) -> dict:
        """Single training step"""
        x = x.to(self.device)
        target = target.to(self.device)
        
        # Forward pass
        output = self.monarch(x)
        loss = F.mse_loss(output, target)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.monarch.parameters(), 1.0)
        
        # Optimizer step
        self.optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            grad_norm = sum(p.grad.norm().item()**2 for p in self.monarch.parameters() if p.grad is not None)**0.5
        
        return {
            'loss': loss.item(),
            'grad_norm': grad_norm,
            'output_norm': output.norm().item()
        }
    
    def get_state_dict(self) -> dict:
        """Get state for checkpointing"""
        return {
            'monarch': self.monarch.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
    
    def load_state_dict(self, state: dict):
        """Load from checkpoint"""
        self.monarch.load_state_dict(state['monarch'])
        self.optimizer.load_state_dict(state['optimizer'])

class MonarchTester:
    """Test suite for Monarch matrices"""
    
    def __init__(self, config: MonarchConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.monarch = MonarchMatrix(config).to(self.device)
    
    def test_complexity(self, n_iterations: int = 100) -> dict:
        """Measure actual complexity"""
        n = self.config.size
        batch_size = 32
        
        # Create test data
        x = torch.randn(batch_size, n, device=self.device)
        
        # Warmup
        for _ in range(10):
            _ = self.monarch(x)
        
        # Time Monarch
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = self.monarch(x)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        monarch_time = time.perf_counter() - start
        
        # Time dense baseline
        dense = nn.Linear(n, n, bias=False).to(self.device)
        
        # Warmup
        for _ in range(10):
            _ = dense(x)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = dense(x)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        dense_time = time.perf_counter() - start
        
        # Compute metrics
        monarch_flops = self.monarch.compute_flops()
        dense_flops = n * n * 2
        
        return {
            'size': n,
            'monarch_time_ms': monarch_time * 1000 / n_iterations,
            'dense_time_ms': dense_time * 1000 / n_iterations,
            'speedup': dense_time / monarch_time,
            'monarch_flops': monarch_flops,
            'dense_flops': dense_flops,
            'flops_reduction': dense_flops / monarch_flops,
            'device': str(self.device)
        }
    
    def test_gradient_flow(self) -> dict:
        """Test gradient flow through Monarch"""
        x = torch.randn(10, self.config.size, device=self.device, requires_grad=True)
        
        # Forward
        y = self.monarch(x)
        
        # Create loss
        target = torch.randn_like(y)
        loss = F.mse_loss(y, target)
        
        # Backward
        loss.backward()
        
        # Check gradients
        input_grad_norm = x.grad.norm().item() if x.grad is not None else 0
        
        param_grad_norms = []
        for name, param in self.monarch.named_parameters():
            if param.grad is not None:
                param_grad_norms.append(param.grad.norm().item())
        
        avg_param_grad = sum(param_grad_norms) / len(param_grad_norms) if param_grad_norms else 0
        
        return {
            'loss': loss.item(),
            'input_grad_norm': input_grad_norm,
            'avg_param_grad_norm': avg_param_grad,
            'max_param_grad_norm': max(param_grad_norms) if param_grad_norms else 0,
            'healthy': 0.0001 < avg_param_grad < 10.0
        }
    
    def test_learning(self, n_steps: int = 100) -> dict:
        """Test that Monarch can learn a simple task"""
        # Create a simple target transformation
        torch.manual_seed(42)
        target_transform = torch.randn(self.config.size, self.config.size, device=self.device) / math.sqrt(self.config.size)
        
        # Optimizer
        optimizer = torch.optim.Adam(self.monarch.parameters(), lr=1e-2)
        
        losses = []
        for step in range(n_steps):
            # Generate batch
            x = torch.randn(32, self.config.size, device=self.device)
            target = x @ target_transform
            
            # Forward
            output = self.monarch(x)
            loss = F.mse_loss(output, target)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        # Check if learning
        initial_loss = sum(losses[:10]) / 10
        final_loss = sum(losses[-10:]) / 10
        
        return {
            'initial_loss': initial_loss,
            'final_loss': final_loss,
            'loss_reduction': (initial_loss - final_loss) / initial_loss,
            'learned': final_loss < initial_loss * 0.5
        }

def run_comprehensive_tests():
    """Run all tests on Monarch implementation"""
    print("="*70)
    print("PURE PYTORCH MONARCH MATRIX TESTS")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print()
    
    # Test different sizes
    sizes = [128, 256, 512, 1024]
    
    for size in sizes:
        print(f"\nSize {size}:")
        print("-"*40)
        
        config = MonarchConfig(size=size, n_blocks=min(16, size//16), device=device)
        tester = MonarchTester(config)
        
        # Complexity test
        complexity = tester.test_complexity(n_iterations=50)
        print(f"  Time speedup: {complexity['speedup']:.2f}x")
        print(f"  FLOPS reduction: {complexity['flops_reduction']:.2f}x")
        print(f"  Monarch: {complexity['monarch_time_ms']:.3f}ms, Dense: {complexity['dense_time_ms']:.3f}ms")
        
        # Gradient test
        grad_test = tester.test_gradient_flow()
        print(f"  Gradients healthy: {'✓' if grad_test['healthy'] else '✗'}")
        print(f"  Avg grad norm: {grad_test['avg_param_grad_norm']:.4f}")
        
        # Learning test
        learning = tester.test_learning(n_steps=50)
        print(f"  Can learn: {'✓' if learning['learned'] else '✗'}")
        print(f"  Loss reduction: {learning['loss_reduction']:.1%}")

def test_ray_actors():
    """Test Ray actor integration"""
    print("\n" + "="*70)
    print("RAY ACTOR TESTS")
    print("="*70)
    
    ray.init(ignore_reinit_error=True, num_gpus=1 if torch.cuda.is_available() else 0)
    
    config = MonarchConfig(size=256, n_blocks=16)
    
    # Create actors
    actors = [MonarchActor.remote(config) for _ in range(2)]
    
    # Test forward pass
    x = torch.randn(10, 256)
    futures = [actor.forward.remote(x) for actor in actors]
    outputs = ray.get(futures)
    
    print(f"Forward pass shapes: {[o.shape for o in outputs]}")
    
    # Test training
    target = torch.randn(10, 256)
    futures = [actor.train_step.remote(x, target) for actor in actors]
    metrics = ray.get(futures)
    
    print(f"Training metrics:")
    for i, m in enumerate(metrics):
        print(f"  Actor {i}: loss={m['loss']:.4f}, grad_norm={m['grad_norm']:.4f}")
    
    ray.shutdown()

if __name__ == "__main__":
    print("PURE PYTORCH MONARCH MATRIX LIBRARY")
    print("="*70)
    
    # Run tests
    run_comprehensive_tests()
    
    # Test Ray actors
    test_ray_actors()
    
    print("\n✓ All tests completed")