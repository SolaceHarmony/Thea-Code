#!/usr/bin/env python3
"""
Liquid Foundation Model (LFM2) Implementation
Based on Poli et al.'s latest work at Liquid AI
Hybrid architecture with multiplicative gating and short convolutions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math
from dataclasses import dataclass


@dataclass
class LFM2Config:
    """Configuration for Liquid Foundation Model v2"""
    dim: int = 768
    n_conv_blocks: int = 10
    n_attention_blocks: int = 6
    conv_kernel_size: int = 3  # Short convolution
    n_heads: int = 12
    n_kv_heads: int = 4  # For GQA
    hidden_dim: int = 3072
    vocab_size: int = 32000
    max_seq_len: int = 32768
    dropout: float = 0.1
    
    # Hardware optimization hints
    target_device: str = "auto"  # "cpu", "gpu", "embedded", "auto"
    optimize_for_latency: bool = True
    
    def get_block_order(self):
        """Get STAR-optimized block ordering"""
        # Interleave conv and attention blocks
        # This is a simplified version - real STAR would optimize this
        order = []
        conv_idx, attn_idx = 0, 0
        
        # Pattern discovered by STAR: mostly conv with strategic attention
        pattern = ['conv'] * 2 + ['attn'] + ['conv'] * 3 + ['attn'] + ['conv'] * 5 + ['attn'] * 4
        
        for block_type in pattern:
            if block_type == 'conv' and conv_idx < self.n_conv_blocks:
                order.append(('conv', conv_idx))
                conv_idx += 1
            elif block_type == 'attn' and attn_idx < self.n_attention_blocks:
                order.append(('attn', attn_idx))
                attn_idx += 1
        
        return order


class LFM2ConvBlock(nn.Module):
    """
    Double-gated short convolution block
    Key innovation: multiplicative gates that depend on input
    """
    
    def __init__(self, config: LFM2Config):
        super().__init__()
        self.dim = config.dim
        
        # Input projection produces x, B gate, C gate
        self.input_proj = nn.Linear(config.dim, config.dim * 3)
        
        # Short convolution (converges to zero after finite time)
        self.conv = nn.Conv1d(
            config.dim, 
            config.dim, 
            kernel_size=config.conv_kernel_size,
            padding=config.conv_kernel_size // 2,
            groups=1  # Could use groups for efficiency
        )
        
        # Output projection
        self.output_proj = nn.Linear(config.dim, config.dim)
        
        # Normalization
        self.norm = nn.RMSNorm(config.dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, dim]
        """
        residual = x
        x = self.norm(x)
        
        # Project to get x and gates
        projected = self.input_proj(x)
        x, B, C = projected.chunk(3, dim=-1)
        
        # First multiplicative gate
        x = B * x  # Element-wise multiplication
        
        # Short convolution
        x = x.transpose(1, 2)  # [batch, dim, seq_len]
        x = self.conv(x)
        x = x.transpose(1, 2)  # [batch, seq_len, dim]
        
        # Second multiplicative gate
        x = C * x
        
        # Output projection
        x = self.output_proj(x)
        
        # Residual connection
        return residual + x


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) - more efficient than standard MHA
    Groups multiple query heads to share key/value heads
    """
    
    def __init__(self, config: LFM2Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.n_groups = config.n_heads // config.n_kv_heads
        
        self.q_proj = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)
        
        self.norm = nn.RMSNorm(config.dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        
        residual = x
        x = self.norm(x)
        
        # Compute Q, K, V
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        
        # Repeat K, V for grouped query attention
        k = k.repeat_interleave(self.n_groups, dim=2)
        v = v.repeat_interleave(self.n_groups, dim=2)
        
        # Transpose for attention
        q = q.transpose(1, 2)  # [batch, n_heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention (use Flash Attention if available)
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False
        )
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch, seq_len, -1)
        attn_output = self.o_proj(attn_output)
        
        return residual + attn_output


class SwiGLU(nn.Module):
    """
    SwiGLU activation - better than standard GELU
    Swish-Gated Linear Unit
    """
    
    def __init__(self, config: LFM2Config):
        super().__init__()
        # Projects to 3x hidden_dim (for gate, up, and down)
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w2 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w3 = nn.Linear(config.hidden_dim, config.dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: x = w3(swish(w1(x)) * w2(x))
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class LFM2Block(nn.Module):
    """
    Complete LFM2 block - either conv or attention based on configuration
    """
    
    def __init__(self, config: LFM2Config, block_type: str):
        super().__init__()
        self.block_type = block_type
        
        if block_type == 'conv':
            self.main = LFM2ConvBlock(config)
        else:  # 'attn'
            self.main = GroupedQueryAttention(config)
        
        # Every block has SwiGLU and RMSNorm
        self.ffn = SwiGLU(config)
        self.ffn_norm = nn.RMSNorm(config.dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Main block (conv or attention)
        if self.block_type == 'attn':
            x = self.main(x, mask)
        else:
            x = self.main(x)
        
        # FFN with residual
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = residual + x
        
        return x


class LiquidFoundationModel(nn.Module):
    """
    Complete Liquid Foundation Model v2
    Hardware-aware, hybrid architecture
    """
    
    def __init__(self, config: LFM2Config):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.dim)
        
        # Position embeddings (RoPE would be better)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.dim)
        
        # Create blocks according to STAR-optimized ordering
        self.blocks = nn.ModuleList()
        for block_type, _ in config.get_block_order():
            self.blocks.append(LFM2Block(config, block_type))
        
        # Output projection
        self.norm = nn.RMSNorm(config.dim)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Optimize for target hardware
        self._optimize_for_hardware()
        
    def _optimize_for_hardware(self):
        """Adapt model based on target hardware"""
        if self.config.target_device == "auto":
            if torch.cuda.is_available():
                self.config.target_device = "gpu"
            elif torch.backends.mps.is_available():
                self.config.target_device = "mps"
            else:
                self.config.target_device = "cpu"
        
        print(f"Optimizing for {self.config.target_device}")
        
        # Hardware-specific optimizations
        if self.config.target_device == "embedded":
            # Quantization for embedded
            self.apply(lambda m: m.half() if isinstance(m, nn.Linear) else m)
        elif self.config.target_device == "gpu":
            # Enable Flash Attention
            torch.backends.cuda.enable_flash_sdp(True)
        
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        x = self.token_emb(input_ids)
        
        # Add position embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = x + self.pos_emb(positions)
        
        # Process through blocks
        for block in self.blocks:
            x = block(x, attention_mask)
        
        # Output
        x = self.norm(x)
        logits = self.output(x)
        
        return logits
    
    def generate(self, prompt_ids: torch.Tensor, max_length: int = 100):
        """Simple generation for testing"""
        self.eval()
        generated = prompt_ids
        
        with torch.no_grad():
            for _ in range(max_length):
                logits = self(generated)
                next_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                generated = torch.cat([generated, next_token], dim=1)
                
                if next_token.item() == 2:  # EOS token
                    break
        
        return generated


def analyze_hardware_efficiency():
    """Analyze efficiency on different hardware"""
    import time
    
    config = LFM2Config(
        dim=512,
        n_conv_blocks=10,
        n_attention_blocks=6,
        max_seq_len=2048
    )
    
    model = LiquidFoundationModel(config)
    
    # Test input
    batch_size = 4
    seq_len = 512
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    model = model.to(device)
    input_ids = input_ids.to(device)
    
    # Warmup
    for _ in range(3):
        _ = model(input_ids)
    
    # Benchmark
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()
    
    start = time.perf_counter()
    for _ in range(10):
        _ = model(input_ids)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()
    
    elapsed = time.perf_counter() - start
    
    print(f"\nPerformance Analysis:")
    print(f"  Device: {device}")
    print(f"  Time per forward: {elapsed/10*1000:.2f}ms")
    print(f"  Throughput: {batch_size * seq_len * 10 / elapsed:.0f} tokens/sec")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    conv_params = sum(p.numel() for name, p in model.named_parameters() if 'conv' in name.lower())
    attn_params = sum(p.numel() for name, p in model.named_parameters() if 'attn' in name.lower() or 'proj' in name.lower())
    
    print(f"\nParameter Distribution:")
    print(f"  Total: {total_params/1e6:.1f}M")
    print(f"  Conv blocks: {conv_params/1e6:.1f}M ({conv_params/total_params*100:.1f}%)")
    print(f"  Attention blocks: {attn_params/1e6:.1f}M ({attn_params/total_params*100:.1f}%)")


if __name__ == "__main__":
    print("="*70)
    print("LIQUID FOUNDATION MODEL v2")
    print("Hardware-aware hybrid architecture")
    print("="*70)
    
    # Test basic functionality
    config = LFM2Config(dim=256, n_conv_blocks=2, n_attention_blocks=1)
    model = LiquidFoundationModel(config)
    
    # Test input
    input_ids = torch.randint(0, config.vocab_size, (2, 128))
    logits = model(input_ids)
    
    print(f"\nModel created successfully!")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Block order: {config.get_block_order()}")
    
    # Analyze efficiency
    analyze_hardware_efficiency()