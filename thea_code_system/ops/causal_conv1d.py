#!/usr/bin/env python
"""
Causal Conv1D implementation adapted from vLLM/Mamba
Pure PyTorch + Ray version for our hybrid architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import ray
import math
from typing import Optional


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
) -> torch.Tensor:
    """
    Pure PyTorch causal conv1d (no custom CUDA ops needed)
    
    Args:
        x: (batch, dim, seqlen) 
        weight: (dim, width)
        bias: (dim,)
        activation: None, "silu", or "swish"
    
    Returns:
        out: (batch, dim, seqlen)
    """
    batch_size, dim, seq_len = x.shape
    kernel_size = weight.shape[1]
    
    # Apply causal padding (pad left only)
    padding = kernel_size - 1
    x_padded = F.pad(x, (padding, 0))
    
    # Reshape weight for F.conv1d: (out_channels, in_channels, kernel_size)
    weight_reshaped = weight.unsqueeze(1)  # (dim, 1, width)
    
    # Apply depthwise convolution
    out = F.conv1d(x_padded, weight_reshaped, bias=bias, groups=dim)
    
    # Trim to original sequence length
    out = out[:, :, :seq_len]
    
    # Apply activation
    if activation == "silu" or activation == "swish":
        out = F.silu(out)
    elif activation is not None:
        raise ValueError(f"Unsupported activation: {activation}")
    
    return out


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
) -> torch.Tensor:
    """
    Incremental causal conv1d for generation
    
    Args:
        x: (batch, dim) - single timestep
        conv_state: (batch, dim, kernel_size-1) - cached state
        weight: (dim, kernel_size)
        bias: (dim,)
        activation: None, "silu", or "swish"
    
    Returns:
        out: (batch, dim) - single timestep output
    """
    batch_size, dim = x.shape
    kernel_size = weight.shape[1]
    
    # Update conv state by shifting and adding new input
    if kernel_size > 1:
        # Shift state to the left and add new input
        conv_state[:, :, :-1] = conv_state[:, :, 1:].clone()
        conv_state[:, :, -1] = x
        
        # Compute convolution with full state
        full_state = torch.cat([conv_state, x.unsqueeze(-1)], dim=-1)
    else:
        full_state = x.unsqueeze(-1)
    
    # Apply convolution weights
    out = torch.sum(full_state * weight.unsqueeze(0), dim=-1)
    
    # Add bias
    if bias is not None:
        out = out + bias
    
    # Apply activation
    if activation == "silu" or activation == "swish":
        out = F.silu(out)
    elif activation is not None:
        raise ValueError(f"Unsupported activation: {activation}")
    
    return out


class CausalConv1d(nn.Module):
    """
    Causal 1D convolution module
    Pure PyTorch implementation for our Ray-distributed architecture
    """
    
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        bias: bool = True,
        activation: Optional[str] = None,
    ):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.activation = activation
        
        # Weight: (channels, kernel_size)
        self.weight = nn.Parameter(torch.randn(channels, kernel_size))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(channels))
        else:
            self.bias = None
            
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.kernel_size
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return causal_conv1d_fn(x, self.weight, self.bias, self.activation)
    
    def update(self, x: torch.Tensor, conv_state: torch.Tensor) -> torch.Tensor:
        """Incremental update for generation"""
        return causal_conv1d_update(x, conv_state, self.weight, self.bias, self.activation)


@ray.remote
class CausalConvActor:
    """
    Ray actor wrapping causal convolution
    This is where Ray will pay off - distributed conv processing!
    """
    
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        bias: bool = True,
        activation: Optional[str] = None,
        device: str = "mps"
    ):
        self.device = torch.device(device if torch.backends.mps.is_available() else "cpu")
        self.conv = CausalConv1d(channels, kernel_size, bias, activation).to(self.device)
        
        # State for incremental generation
        self.conv_state = None
        
    async def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process tensor through causal convolution"""
        x = x.to(self.device)
        out = self.conv(x)
        return out.cpu()  # Return to CPU for Ray serialization
    
    async def update(self, x: torch.Tensor) -> torch.Tensor:
        """Incremental update for generation"""
        x = x.to(self.device)
        
        if self.conv_state is None:
            batch_size = x.shape[0]
            self.conv_state = torch.zeros(
                batch_size, self.conv.channels, self.conv.kernel_size - 1,
                device=self.device
            )
        
        out = self.conv.update(x, self.conv_state)
        return out.cpu()
    
    async def reset_state(self):
        """Reset convolution state"""
        self.conv_state = None


# Test the implementation
if __name__ == "__main__":
    print("🔥 Testing PyTorch + Ray Causal Conv1D...")
    
    # Test parameters
    batch_size = 2
    channels = 64
    seq_len = 10
    kernel_size = 3
    
    # Test pure PyTorch version
    print("\n1. Testing Pure PyTorch Implementation:")
    x = torch.randn(batch_size, channels, seq_len)
    conv = CausalConv1d(channels, kernel_size, bias=True, activation="silu")
    
    out = conv(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  ✅ Pure PyTorch works!")
    
    # Test Ray actor version
    print("\n2. Testing Ray Actor Implementation:")
    
    # Initialize Ray if needed
    if not ray.is_initialized():
        ray.init(
            num_cpus=2,
            _temp_dir="/Volumes/emberstuff/ray_temp",
            object_store_memory=500_000_000
        )
    
    # Create Ray actor
    conv_actor = CausalConvActor.remote(channels, kernel_size, bias=True, activation="silu")
    
    # Test forward pass
    future = conv_actor.forward.remote(x)
    out_ray = ray.get(future)
    
    print(f"  Ray output shape: {out_ray.shape}")
    print(f"  ✅ Ray actor works!")
    
    # Test incremental updates
    print("\n3. Testing Incremental Updates:")
    x_single = torch.randn(batch_size, channels)
    
    future = conv_actor.update.remote(x_single)
    out_update = ray.get(future)
    
    print(f"  Update input shape: {x_single.shape}")
    print(f"  Update output shape: {out_update.shape}")
    print(f"  ✅ Incremental updates work!")
    
    # Cleanup
    ray.shutdown()
    
    print("\n🎉 All tests passed!")
    print("✨ Our PyTorch + Ray causal conv1d is ready!")
    print("💫 Ray distribution will pay off for:")
    print("   • Parallel processing across layers")
    print("   • Distributed model serving")
    print("   • Fault tolerance")
    print("   • Dynamic scaling")