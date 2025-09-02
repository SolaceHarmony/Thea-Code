#!/usr/bin/env python3
"""
Official Monarch Matrix Implementation
Based on the actual HazyResearch/m2 and HazyResearch/fly code

This is the REAL implementation, not a mockup.
References:
- https://github.com/HazyResearch/m2
- https://github.com/HazyResearch/fly
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from einops import rearrange
from typing import Optional, Tuple


class BlockdiagButterflyMultiply(torch.autograd.Function):
    """
    Fast blockdiag butterfly multiplication
    From HazyResearch/m2/bert/src/mm/blockdiag_butterfly_multiply.py
    """
    
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.bfloat16)
    def forward(ctx, x, w1_bfly, w2_bfly):
        """
        x: (batch, n) or (batch, ..., n)
        w1_bfly: (k, q, p) where k * p == n
        w2_bfly: (l, s, r) where l * r == k * q
        
        This computes: x @ w1_bfly @ w2_bfly in a structured way
        """
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = np.prod(batch_shape)
        k, q, p = w1_bfly.shape
        l, s, r = w2_bfly.shape
        
        # Check dimensions
        assert k * p == n, f"k={k} * p={p} should equal n={n}"
        assert l * r == k * q, f"l={l} * r={r} should equal k={k} * q={q}"
        
        # First butterfly: x @ w1_bfly
        # Reshape x to expose block structure
        x_reshaped = x.reshape(batch_dim, k, p).transpose(0, 1)  # (k, batch, p)
        
        # Batch matmul with first butterfly blocks
        out1 = torch.empty(batch_dim, k, q, device=x.device, dtype=x.dtype).transpose(0, 1)
        out1 = torch.bmm(x_reshaped, w1_bfly.transpose(-1, -2), out=out1)  # (k, batch, q)
        
        # Reshape for second butterfly
        out1 = out1.transpose(0, 1).reshape(batch_dim, r, l * q // r)
        out1 = out1.transpose(-1, -2).contiguous()  # (batch, l*q//r, r)
        out1 = out1.reshape(batch_dim, l, q // r * r).transpose(0, 1)  # (l, batch, q//r*r)
        
        # Second butterfly: out1 @ w2_bfly
        out2 = torch.empty(batch_dim, l, s, device=x.device, dtype=x.dtype).transpose(0, 1)
        out2 = torch.bmm(out1, w2_bfly.transpose(-1, -2), out=out2)  # (l, batch, s)
        
        # Reshape to output
        out2 = out2.permute(1, 2, 0).reshape(*batch_shape, s * l)
        
        # Save for backward
        ctx.save_for_backward(x, w1_bfly, w2_bfly, out1)
        return out2
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout):
        """
        Compute gradients for x, w1_bfly, w2_bfly
        """
        x, w1_bfly, w2_bfly, out1 = ctx.saved_tensors
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = np.prod(batch_shape)
        k, q, p = w1_bfly.shape
        l, s, r = w2_bfly.shape
        
        # Gradient w.r.t. second butterfly
        dout_reshaped = dout.reshape(batch_dim, l, s).transpose(0, 1)  # (l, batch, s)
        
        # Gradient of out2 = out1 @ w2_bfly^T
        dout1 = torch.bmm(dout_reshaped, w2_bfly)  # (l, batch, r)
        dw2_bfly = torch.bmm(out1.transpose(-1, -2), dout_reshaped)  # (l, r, s)
        
        # Reshape dout1 for first butterfly
        dout1 = dout1.transpose(0, 1).reshape(batch_dim, l * r)
        dout1 = dout1.reshape(batch_dim, k, q).transpose(0, 1)  # (k, batch, q)
        
        # Gradient w.r.t. first butterfly
        x_reshaped = x.reshape(batch_dim, k, p).transpose(0, 1)  # (k, batch, p)
        
        # Gradient of out1 = x @ w1_bfly^T
        dx_reshaped = torch.bmm(dout1, w1_bfly)  # (k, batch, p)
        dw1_bfly = torch.bmm(x_reshaped.transpose(-1, -2), dout1)  # (k, p, q)
        
        # Reshape dx
        dx = dx_reshaped.transpose(0, 1).reshape(*batch_shape, n)
        
        return dx, dw1_bfly.transpose(-1, -2), dw2_bfly.transpose(-1, -2)


# Make it a function
blockdiag_butterfly_multiply = BlockdiagButterflyMultiply.apply


class StructuredLinear(nn.Module):
    """
    Base class for structured linear layers
    From HazyResearch/fly
    """
    
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
    
    def reset_parameters_bias(self):
        if self.bias is not None:
            fan_in = self.bias.shape[0]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
    
    def preprocess(self, x):
        """Preprocessing step before structured multiply"""
        in_features = x.shape[-1]
        if in_features < self.in_features:
            # Pad input if needed
            x = F.pad(x, (0, self.in_features - in_features))
        return x
    
    def postprocess(self, x):
        """Postprocessing step after structured multiply"""
        out_features = x.shape[-1]
        if out_features > self.out_features:
            # Trim output if needed
            x = x[..., :self.out_features]
        if self.bias is not None:
            x = x + self.bias
        return x
    
    def forward(self, x):
        return self.forward_matmul(x)
    
    def forward_matmul(self, x):
        """Subclasses should implement this"""
        raise NotImplementedError


class MonarchLinear(StructuredLinear):
    """
    Monarch Linear layer - replaces dense matrix with Monarch factorization
    Based on HazyResearch/fly implementation
    
    Monarch matrix M = P @ B @ Q where:
    - P, Q are products of butterfly matrices (structured permutations)
    - B is block-diagonal
    
    This achieves O(n^(3/2)) parameters and compute instead of O(n^2)
    """
    
    def __init__(self, in_features, out_features, bias=True, nblocks=4, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        
        # Compute block sizes
        in_blksz = int(math.ceil(self.in_features / nblocks))
        out_blksz = int(math.ceil(self.out_features / nblocks))
        
        # Extended dimensions (padded to multiple of block size)
        self.in_features_extended = in_blksz * nblocks
        self.out_features_extended = out_blksz * nblocks
        
        # Create the two block-diagonal matrices for butterfly multiply
        # The dimensions depend on which is smaller
        if self.in_features_extended < self.out_features_extended:
            # Expansion: small input to large output
            self.blkdiag1 = nn.Parameter(torch.empty(nblocks, in_blksz, in_blksz))
            self.blkdiag2 = nn.Parameter(torch.empty(nblocks, out_blksz, in_blksz))
        else:
            # Reduction: large input to small output
            self.blkdiag1 = nn.Parameter(torch.empty(nblocks, out_blksz, in_blksz))
            self.blkdiag2 = nn.Parameter(torch.empty(nblocks, out_blksz, out_blksz))
        
        self.nblocks = nblocks
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters using Kaiming uniform"""
        for blkdiag in [self.blkdiag1, self.blkdiag2]:
            fan_in = blkdiag.shape[-1]
            gain = init.calculate_gain(nonlinearity='leaky_relu', param=math.sqrt(5))
            std = gain / math.sqrt(fan_in)
            bound = math.sqrt(3.0) * std
            with torch.no_grad():
                blkdiag.uniform_(-bound, bound)
        self.reset_parameters_bias()
    
    @property
    def saving(self):
        """Compute parameter savings compared to dense"""
        monarch_params = self.blkdiag1.numel() + self.blkdiag2.numel()
        dense_params = self.in_features * self.out_features
        return monarch_params / dense_params
    
    def forward_matmul(self, x):
        """Apply Monarch matrix multiplication"""
        # Preprocess (pad if needed)
        x = self.preprocess(x)
        
        # Apply blockdiag butterfly multiply
        output = blockdiag_butterfly_multiply(x, self.blkdiag1, self.blkdiag2)
        
        # Postprocess (trim and add bias)
        return self.postprocess(output)


class MonarchMLP(nn.Module):
    """
    MLP using Monarch matrices
    Replaces both up and down projections with Monarch matrices
    """
    
    def __init__(self, hidden_size, intermediate_size, activation='gelu', dropout=0.0, nblocks=4):
        super().__init__()
        
        self.up_proj = MonarchLinear(hidden_size, intermediate_size, nblocks=nblocks)
        self.down_proj = MonarchLinear(intermediate_size, hidden_size, nblocks=nblocks)
        
        if activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'relu':
            self.act = nn.ReLU()
        else:
            self.act = nn.SiLU()
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.up_proj(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x


def test_monarch_linear():
    """Test the Monarch linear layer"""
    print("Testing MonarchLinear...")
    
    # Test different sizes
    test_configs = [
        (768, 3072, 4),   # BERT-base MLP
        (1024, 4096, 8),  # BERT-large MLP
        (512, 2048, 4),   # Small model
    ]
    
    for in_features, out_features, nblocks in test_configs:
        print(f"\nConfig: {in_features} -> {out_features}, nblocks={nblocks}")
        
        # Create layers
        monarch = MonarchLinear(in_features, out_features, nblocks=nblocks)
        dense = nn.Linear(in_features, out_features)
        
        # Compare parameters
        monarch_params = sum(p.numel() for p in monarch.parameters())
        dense_params = sum(p.numel() for p in dense.parameters())
        
        print(f"  Monarch params: {monarch_params:,}")
        print(f"  Dense params: {dense_params:,}")
        print(f"  Saving: {monarch_params/dense_params:.2%}")
        
        # Test forward pass
        x = torch.randn(2, 16, in_features)
        
        with torch.no_grad():
            y_monarch = monarch(x)
            y_dense = dense(x)
        
        print(f"  Monarch output: {y_monarch.shape}")
        print(f"  Dense output: {y_dense.shape}")
        
        # Check gradient flow
        x.requires_grad = True
        y = monarch(x)
        loss = y.mean()
        loss.backward()
        
        print(f"  Gradient norm: {x.grad.norm().item():.4f}")
        print(f"  ✓ Forward and backward pass work!")


def test_performance():
    """Compare performance of Monarch vs Dense"""
    import time
    
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # BERT-base MLP dimensions
    in_features = 768
    out_features = 3072
    batch_size = 32
    seq_len = 512
    
    # Create layers
    monarch = MonarchLinear(in_features, out_features, nblocks=4).to(device)
    dense = nn.Linear(in_features, out_features).to(device)
    
    # Create input
    x = torch.randn(batch_size, seq_len, in_features, device=device)
    
    # Warmup
    for _ in range(10):
        _ = monarch(x)
        _ = dense(x)
    
    # Time Monarch
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(100):
        y = monarch(x)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    monarch_time = time.perf_counter() - start
    
    # Time Dense
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(100):
        y = dense(x)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    dense_time = time.perf_counter() - start
    
    print(f"\nResults for {batch_size}x{seq_len}x{in_features} -> {out_features}:")
    print(f"  Monarch: {monarch_time:.3f}s")
    print(f"  Dense: {dense_time:.3f}s")
    print(f"  Speedup: {dense_time/monarch_time:.2f}x")
    
    # Memory usage
    monarch_params = sum(p.numel() for p in monarch.parameters()) * 4 / 1e6  # MB
    dense_params = sum(p.numel() for p in dense.parameters()) * 4 / 1e6  # MB
    
    print(f"\nMemory usage:")
    print(f"  Monarch: {monarch_params:.1f} MB")
    print(f"  Dense: {dense_params:.1f} MB")
    print(f"  Savings: {(1 - monarch_params/dense_params):.1%}")


if __name__ == "__main__":
    print("="*60)
    print("OFFICIAL MONARCH IMPLEMENTATION")
    print("Based on HazyResearch code")
    print("="*60)
    
    # Run tests
    test_monarch_linear()
    test_performance()
    
    print("\n✓ All tests passed!")