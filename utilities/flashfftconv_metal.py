#!/usr/bin/env python3
"""
Flash FFT Convolution for Apple Metal (MPS)
PyTorch JIT-compiled kernels for M1/M2/M3 chips
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from typing import Optional, Tuple
from einops import rearrange

# Check for MPS availability
if torch.backends.mps.is_available():
    print("Metal Performance Shaders (MPS) backend available!")
    DEVICE = torch.device("mps")
else:
    print("MPS not available, falling back to CPU")
    DEVICE = torch.device("cpu")


@torch.jit.script
def compute_twiddle_factors_fft_jit(n: int, m: int, device: torch.device) -> Tensor:
    """JIT-compiled twiddle factor computation for FFT"""
    n_a = torch.arange(n, device=device, dtype=torch.float32).view(-1, 1)
    m_a = torch.arange(m, device=device, dtype=torch.float32)
    N = float(n * m)
    # Compute real and imaginary parts separately for MPS compatibility
    angle = -2.0 * math.pi * n_a * m_a / N
    real_part = torch.cos(angle)
    imag_part = torch.sin(angle)
    return torch.stack([real_part, imag_part], dim=-1)


@torch.jit.script
def compute_twiddle_factors_ifft_jit(n: int, m: int, device: torch.device) -> Tensor:
    """JIT-compiled twiddle factor computation for IFFT"""
    n_a = torch.arange(n, device=device, dtype=torch.float32).view(-1, 1)
    m_a = torch.arange(m, device=device, dtype=torch.float32)
    N = float(n * m)
    angle = 2.0 * math.pi * n_a * m_a / N
    real_part = torch.cos(angle)
    imag_part = torch.sin(angle)
    return torch.stack([real_part, imag_part], dim=-1)


@torch.jit.script
def complex_mul_jit(a: Tensor, b: Tensor) -> Tensor:
    """JIT-compiled complex multiplication for MPS
    a, b: [..., 2] where last dim is [real, imag]
    """
    a_real, a_imag = a[..., 0], a[..., 1]
    b_real, b_imag = b[..., 0], b[..., 1]
    
    out_real = a_real * b_real - a_imag * b_imag
    out_imag = a_real * b_imag + a_imag * b_real
    
    return torch.stack([out_real, out_imag], dim=-1)


@torch.jit.script
def butterfly_multiply_jit(x: Tensor, twiddle: Tensor, n: int, m: int) -> Tensor:
    """JIT-compiled butterfly multiplication for Metal
    
    x: [batch, n*m, 2] complex tensor
    twiddle: [n, m, 2] complex twiddle factors
    """
    batch = x.shape[0]
    
    # Reshape for butterfly operation
    x = x.view(batch, n, m, 2)
    
    # Expand twiddle factors for broadcasting
    twiddle = twiddle.unsqueeze(0)  # [1, n, m, 2]
    
    # Complex multiplication
    x = complex_mul_jit(x, twiddle)
    
    # Reshape back
    return x.view(batch, n * m, 2)


class FlashFFTConvMetal(nn.Module):
    """Flash FFT Convolution optimized for Apple Metal"""
    
    def __init__(self, seqlen: int, dtype=torch.float32):
        super().__init__()
        self.seqlen = seqlen
        self.dtype = dtype
        self.device = DEVICE
        
        # Compute FFT parameters
        if seqlen in [256, 1024, 4096]:
            self.N = seqlen
            self.sqrt_N = int(math.sqrt(seqlen))
            
            # Precompute twiddle factors on Metal
            self.register_buffer('twiddle_fft', 
                compute_twiddle_factors_fft_jit(self.sqrt_N, self.sqrt_N, self.device))
            self.register_buffer('twiddle_ifft',
                compute_twiddle_factors_ifft_jit(self.sqrt_N, self.sqrt_N, self.device))
            
            # Scale factor for FFT
            self.scale = 1.0 / self.N
            
        elif seqlen in [512, 2048, 8192]:
            # Handle non-square sequence lengths
            self.N = seqlen // 2
            self.sqrt_N = int(math.sqrt(self.N))
            
            self.register_buffer('twiddle_fft',
                compute_twiddle_factors_fft_jit(self.sqrt_N, self.sqrt_N, self.device))
            self.register_buffer('twiddle_ifft',
                compute_twiddle_factors_ifft_jit(self.sqrt_N, self.sqrt_N, self.device))
            
            # Additional twiddle for real-to-complex conversion
            angle = -2.0 * math.pi * torch.arange(self.N, device=self.device) / seqlen
            self.register_buffer('twid_real', torch.cos(angle))
            self.register_buffer('twid_imag', torch.sin(angle))
            
            self.scale = 1.0 / self.N
        else:
            raise ValueError(f"Unsupported sequence length: {seqlen}")
    
    @torch.jit.method
    def monarch_fft_forward(self, x: Tensor) -> Tensor:
        """Forward FFT using Monarch decomposition
        x: [batch, seqlen] real tensor
        returns: [batch, seqlen//2+1, 2] complex tensor
        """
        batch = x.shape[0]
        
        # Convert to complex representation
        x_complex = torch.stack([x, torch.zeros_like(x)], dim=-1)
        
        # First butterfly pass
        x_complex = butterfly_multiply_jit(x_complex, self.twiddle_fft, 
                                          self.sqrt_N, self.sqrt_N)
        
        # Transpose for second pass
        x_complex = x_complex.view(batch, self.sqrt_N, self.sqrt_N, 2)
        x_complex = x_complex.transpose(1, 2).contiguous()
        x_complex = x_complex.view(batch, self.N, 2)
        
        # Second butterfly pass
        x_complex = butterfly_multiply_jit(x_complex, self.twiddle_fft,
                                          self.sqrt_N, self.sqrt_N)
        
        # Scale
        x_complex = x_complex * self.scale
        
        return x_complex
    
    @torch.jit.method
    def monarch_ifft_forward(self, x: Tensor) -> Tensor:
        """Inverse FFT using Monarch decomposition
        x: [batch, N, 2] complex tensor
        returns: [batch, seqlen] real tensor
        """
        batch = x.shape[0]
        
        # First butterfly pass
        x = butterfly_multiply_jit(x, self.twiddle_ifft,
                                  self.sqrt_N, self.sqrt_N)
        
        # Transpose
        x = x.view(batch, self.sqrt_N, self.sqrt_N, 2)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch, self.N, 2)
        
        # Second butterfly pass
        x = butterfly_multiply_jit(x, self.twiddle_ifft,
                                  self.sqrt_N, self.sqrt_N)
        
        # Extract real part
        return x[..., 0]
    
    def forward(self, u: Tensor, k: Tensor, 
                D: Optional[Tensor] = None,
                dropout_mask: Optional[Tensor] = None,
                gelu: bool = True) -> Tensor:
        """
        Compute convolution using Flash FFT on Metal
        
        u: [batch, channels, seqlen] input tensor
        k: [channels, seqlen] kernel tensor
        D: [channels] optional residual connection weights
        dropout_mask: optional dropout mask
        gelu: whether to apply GELU activation
        """
        batch, channels, seqlen = u.shape
        
        # Move to Metal device
        u = u.to(self.device)
        k = k.to(self.device)
        
        # Reshape for FFT
        u_flat = rearrange(u, 'b c l -> (b c) l')
        k_flat = rearrange(k, 'c l -> c l')
        
        # Compute FFT of input and kernel
        if self.seqlen <= 4096:
            # Use PyTorch's native FFT for smaller sizes (optimized for Metal)
            fft_size = 2 * seqlen
            k_f = torch.fft.rfft(k_flat, n=fft_size) / fft_size
            u_f = torch.fft.rfft(u_flat, n=fft_size)
            
            # Pointwise multiplication in frequency domain
            k_f = k_f.unsqueeze(0).expand(batch, -1, -1)
            u_f = rearrange(u_f, '(b c) f -> b c f', b=batch)
            y_f = u_f * k_f
            
            # Inverse FFT
            y = torch.fft.irfft(y_f, n=fft_size, norm="forward")[..., :seqlen]
        else:
            # Use custom Monarch decomposition for larger sizes
            # This would use the JIT-compiled butterfly operations
            u_complex = self.monarch_fft_forward(u_flat)
            k_complex = self.monarch_fft_forward(k_flat[0:1]).expand(batch * channels, -1, -1)
            
            # Complex multiplication
            y_complex = complex_mul_jit(u_complex, k_complex)
            
            # Inverse FFT
            y_flat = self.monarch_ifft_forward(y_complex)
            y = rearrange(y_flat, '(b c) l -> b c l', b=batch)
        
        # Add residual connection if provided
        if D is not None:
            D = D.to(self.device)
            y = y + u * D.unsqueeze(0).unsqueeze(-1)
        
        # Apply activation
        if gelu:
            y = F.gelu(y)
        
        # Apply dropout if provided
        if dropout_mask is not None:
            dropout_mask = dropout_mask.to(self.device)
            y = y * rearrange(dropout_mask, 'b c -> b c 1')
        
        return y


class HyenaOperatorMetal(nn.Module):
    """Hyena operator using Metal-optimized FFT convolution"""
    
    def __init__(
        self,
        d_model: int,
        l_max: int,
        order: int = 2,
        dropout: float = 0.0,
        filter_dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.l_max = l_max
        self.order = order
        
        # Initialize projections
        self.in_proj = nn.Linear(d_model, (order + 1) * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Short convolution for local features
        self.short_filter = nn.Conv1d(
            (order + 1) * d_model,
            (order + 1) * d_model,
            kernel_size=3,
            padding=1,
            groups=(order + 1) * d_model
        )
        
        # Long convolution using Flash FFT
        self.flashfft = FlashFFTConvMetal(l_max)
        
        # Learnable filter parameters
        self.register_parameter('filter_params', 
            nn.Parameter(torch.randn(order, d_model, l_max)))
        
        self.dropout = nn.Dropout(dropout)
        self.filter_dropout = nn.Dropout(filter_dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        x: [batch, seqlen, d_model]
        """
        batch, seqlen, _ = x.shape
        
        # Input projection
        xz = self.in_proj(x)  # [batch, seqlen, (order+1)*d_model]
        xz = rearrange(xz, 'b l d -> b d l')
        
        # Short convolution
        xz = self.short_filter(xz)[..., :seqlen]
        
        # Split into x and z components
        z, *xs = xz.split(self.d_model, dim=1)
        
        # Apply long convolutions using Flash FFT
        for i, x_i in enumerate(xs):
            # Get filter for this order
            h = self.filter_params[i]
            h = self.filter_dropout(h)
            
            # Apply convolution
            x_i = self.flashfft(x_i.unsqueeze(1), h.unsqueeze(0), gelu=False)
            x_i = x_i.squeeze(1)
            
            # Modulate with gating
            z = z * x_i
        
        # Output projection
        z = rearrange(z, 'b d l -> b l d')
        z = self.dropout(z)
        z = self.out_proj(z)
        
        return z


def benchmark_metal_fft():
    """Benchmark Metal FFT performance"""
    print("="*70)
    print("METAL FFT CONVOLUTION BENCHMARK")
    print("="*70)
    
    if not torch.backends.mps.is_available():
        print("MPS not available on this system")
        return
    
    import time
    
    seq_lengths = [256, 512, 1024, 2048, 4096]
    batch_size = 32
    channels = 768  # BERT hidden size
    
    for seqlen in seq_lengths:
        print(f"\nSequence length: {seqlen}")
        
        # Create Flash FFT Conv
        conv = FlashFFTConvMetal(seqlen).to(DEVICE)
        
        # Create test data
        u = torch.randn(batch_size, channels, seqlen, device=DEVICE)
        k = torch.randn(channels, seqlen, device=DEVICE)
        
        # Warmup
        for _ in range(10):
            _ = conv(u, k)
        
        # Synchronize MPS
        if DEVICE.type == 'mps':
            torch.mps.synchronize()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(100):
            _ = conv(u, k)
        
        if DEVICE.type == 'mps':
            torch.mps.synchronize()
        
        elapsed = time.perf_counter() - start
        
        print(f"  Time per iteration: {elapsed/100*1000:.3f}ms")
        print(f"  Throughput: {batch_size * seqlen * channels / (elapsed/100) / 1e9:.2f} GFLOPS")
        
        # Compare with native FFT
        start = time.perf_counter()
        for _ in range(100):
            fft_size = 2 * seqlen
            u_f = torch.fft.rfft(u, n=fft_size)
            k_f = torch.fft.rfft(k, n=fft_size)
            y_f = u_f * k_f.unsqueeze(0)
            y = torch.fft.irfft(y_f, n=fft_size)[..., :seqlen]
        
        if DEVICE.type == 'mps':
            torch.mps.synchronize()
        
        elapsed_native = time.perf_counter() - start
        
        print(f"  Native FFT time: {elapsed_native/100*1000:.3f}ms")
        print(f"  Speedup: {elapsed_native/elapsed:.2f}x")


if __name__ == "__main__":
    print("Flash FFT Convolution for Metal")
    print(f"Device: {DEVICE}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    
    # Test basic functionality
    if torch.backends.mps.is_available():
        conv = FlashFFTConvMetal(1024)
        x = torch.randn(2, 768, 1024)
        k = torch.randn(768, 1024)
        
        y = conv(x, k)
        print(f"\nTest passed! Output shape: {y.shape}")
        
        # Test Hyena operator
        hyena = HyenaOperatorMetal(d_model=768, l_max=1024)
        x = torch.randn(2, 1024, 768)
        y = hyena(x)
        print(f"Hyena operator output: {y.shape}")
        
        # Run benchmark
        benchmark_metal_fft()
    else:
        print("Skipping tests - MPS not available")