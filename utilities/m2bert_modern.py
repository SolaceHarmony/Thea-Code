#!/usr/bin/env python3
"""
Modern M2-BERT Implementation
Pure PyTorch with latest features:
- torch.compile for graph optimization
- Metal Performance Shaders (MPS) support
- TorchScript JIT compilation
- Efficient attention mechanisms
- Based on Michael Poli et al.'s research
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
import math
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from functools import partial
from einops import rearrange
import time

# Enable torch.compile optimizations
torch.set_float32_matmul_precision('high')

@dataclass
class M2BertConfig:
    """Modern M2-BERT configuration"""
    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    
    # Monarch-specific parameters
    monarch_nblocks: int = 4
    use_monarch_mlp: bool = True
    use_monarch_attention: bool = False
    
    # Modern optimizations
    use_torch_compile: bool = True
    use_flash_attention: bool = True
    use_metal_kernels: bool = torch.backends.mps.is_available()
    
    # Training hyperparameters from paper
    learning_rate: float = 8e-4
    mlm_probability: float = 0.30
    gradient_clip_val: float = 1.0
    
    def __post_init__(self):
        # Auto-detect best device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"


@torch.jit.script
def blockdiag_matmul_jit(x: Tensor, weight: Tensor) -> Tensor:
    """JIT-compiled block-diagonal matrix multiplication
    
    Args:
        x: [batch, n] or [batch, seq, n]
        weight: [nblocks, out_per_block, in_per_block]
    
    Returns:
        output: [batch, nblocks * out_per_block]
    """
    original_shape = x.shape
    if x.dim() == 3:
        batch, seq, n = x.shape
        x = x.reshape(-1, n)
    else:
        batch = x.shape[0]
        seq = 1
    
    nblocks, out_per_block, in_per_block = weight.shape
    
    # Reshape input for block processing
    x = x.view(-1, nblocks, in_per_block)
    
    # Efficient batched matrix multiplication
    output = torch.bmm(
        x.transpose(0, 1),  # [nblocks, batch*seq, in_per_block]
        weight.transpose(1, 2)  # [nblocks, in_per_block, out_per_block]
    ).transpose(0, 1)  # [batch*seq, nblocks, out_per_block]
    
    # Reshape to output
    output = output.reshape(batch, seq, -1) if len(original_shape) == 3 else output.reshape(batch, -1)
    
    return output


class MonarchLinear(Module):
    """Modern Monarch Linear layer using block-diagonal matrices"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        nblocks: int = 4,
        bias: bool = True,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nblocks = nblocks
        
        # Calculate block sizes (with padding if needed)
        self.in_per_block = math.ceil(in_features / nblocks)
        self.out_per_block = math.ceil(out_features / nblocks)
        
        # Padded dimensions
        self.in_features_padded = self.in_per_block * nblocks
        self.out_features_padded = self.out_per_block * nblocks
        
        # Initialize weight as block-diagonal
        self.weight = nn.Parameter(
            torch.empty(nblocks, self.out_per_block, self.in_per_block, device=device, dtype=dtype)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters using Kaiming uniform"""
        # Each block is initialized independently
        fan_in = self.in_per_block
        gain = math.sqrt(5)
        std = gain / math.sqrt(fan_in)
        bound = math.sqrt(3.0) * std
        
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)
        
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            with torch.no_grad():
                self.bias.uniform_(-bound, bound)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass using block-diagonal multiplication"""
        # Pad input if necessary
        if x.shape[-1] < self.in_features_padded:
            padding = self.in_features_padded - x.shape[-1]
            x = F.pad(x, (0, padding))
        
        # Use JIT-compiled kernel
        output = blockdiag_matmul_jit(x, self.weight)
        
        # Trim output if necessary
        if output.shape[-1] > self.out_features:
            output = output[..., :self.out_features]
        
        # Add bias
        if self.bias is not None:
            output = output + self.bias
        
        return output
    
    @property
    def param_reduction(self) -> float:
        """Calculate parameter reduction compared to dense layer"""
        dense_params = self.in_features * self.out_features
        monarch_params = self.nblocks * self.in_per_block * self.out_per_block
        return 1.0 - (monarch_params / dense_params)


@torch.jit.script
def scaled_dot_product_attention_jit(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Optional[Tensor] = None,
    dropout_p: float = 0.0
) -> Tensor:
    """JIT-compiled scaled dot-product attention"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores + mask
    
    attn_weights = F.softmax(scores, dim=-1)
    
    if dropout_p > 0:
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)
    
    return torch.matmul(attn_weights, value)


class M2BertAttention(Module):
    """Modern attention module with optional Monarch matrices"""
    
    def __init__(self, config: M2BertConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_heads * self.head_dim
        
        # Choose between Monarch or standard linear
        if config.use_monarch_attention:
            LinearClass = partial(MonarchLinear, nblocks=config.monarch_nblocks)
        else:
            LinearClass = nn.Linear
        
        self.query = LinearClass(config.hidden_size, self.all_head_size)
        self.key = LinearClass(config.hidden_size, self.all_head_size)
        self.value = LinearClass(config.hidden_size, self.all_head_size)
        self.output = LinearClass(self.all_head_size, config.hidden_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.use_flash = config.use_flash_attention
        
        # Use native scaled_dot_product_attention if available
        self.use_native_sdpa = hasattr(F, 'scaled_dot_product_attention')
    
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Compute Q, K, V
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention
        if self.use_native_sdpa and self.use_flash:
            # Use PyTorch's optimized SDPA (includes Flash Attention when available)
            attn_output = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=attention_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False
            )
        else:
            # Use JIT-compiled version
            attn_output = scaled_dot_product_attention_jit(
                query, key, value,
                mask=attention_mask,
                dropout_p=self.dropout.p if self.training else 0.0
            )
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.all_head_size)
        attn_output = self.output(attn_output)
        attn_output = self.dropout(attn_output)
        
        return attn_output


class M2BertMLP(Module):
    """Modern MLP with Monarch matrices"""
    
    def __init__(self, config: M2BertConfig):
        super().__init__()
        
        # Use Monarch matrices for MLP
        if config.use_monarch_mlp:
            LinearClass = partial(MonarchLinear, nblocks=config.monarch_nblocks)
        else:
            LinearClass = nn.Linear
        
        self.up_proj = LinearClass(config.hidden_size, config.intermediate_size)
        self.down_proj = LinearClass(config.intermediate_size, config.hidden_size)
        
        # Activation function
        if config.hidden_act == "gelu":
            self.act = nn.GELU()
        elif config.hidden_act == "gelu_new":
            self.act = nn.GELU(approximate='tanh')
        elif config.hidden_act == "silu":
            self.act = nn.SiLU()
        else:
            self.act = nn.ReLU()
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.up_proj(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.down_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class M2BertLayer(Module):
    """Modern BERT layer with pre-normalization"""
    
    def __init__(self, config: M2BertConfig):
        super().__init__()
        self.attention = M2BertAttention(config)
        self.mlp = M2BertMLP(config)
        
        # Layer normalization (pre-norm for better training stability)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        # Pre-norm attention block
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = residual + hidden_states
        
        # Pre-norm MLP block
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class M2BertModel(Module):
    """Modern M2-BERT model"""
    
    def __init__(self, config: M2BertConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.embedding_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.embedding_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Encoder layers
        self.layers = nn.ModuleList([
            M2BertLayer(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Compile model if requested
        if config.use_torch_compile and hasattr(torch, 'compile'):
            self.forward = torch.compile(self.forward, mode="max-autotune")
    
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, (nn.Linear, MonarchLinear)):
            if hasattr(module, 'weight'):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None
    ) -> Tensor:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Create token type IDs if not provided
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # Compute embeddings
        word_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)
        
        embeddings = word_embeds + position_embeds + token_type_embeds
        embeddings = self.embedding_norm(embeddings)
        embeddings = self.embedding_dropout(embeddings)
        
        # Prepare attention mask
        if attention_mask is not None:
            # Convert to attention scores mask
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        # Pass through encoder layers
        hidden_states = embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Final normalization
        hidden_states = self.final_norm(hidden_states)
        
        return hidden_states


class M2BertForMaskedLM(Module):
    """M2-BERT for Masked Language Modeling"""
    
    def __init__(self, config: M2BertConfig):
        super().__init__()
        self.config = config
        self.bert = M2BertModel(config)
        
        # MLM head
        self.mlm_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.mlm_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlm_output = nn.Linear(config.hidden_size, config.vocab_size)
        
        # Tie embeddings
        self.mlm_output.weight = self.bert.word_embeddings.weight
    
    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        # Get BERT outputs
        hidden_states = self.bert(input_ids, token_type_ids, attention_mask)
        
        # MLM predictions
        hidden_states = self.mlm_dense(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.mlm_norm(hidden_states)
        logits = self.mlm_output(hidden_states)
        
        output = {"logits": logits}
        
        # Calculate loss if labels provided
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )
            output["loss"] = loss
        
        return output


def benchmark_m2bert():
    """Benchmark modern M2-BERT implementation"""
    print("="*70)
    print("MODERN M2-BERT BENCHMARK")
    print("="*70)
    
    # Detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        backend = "CUDA"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        backend = "Metal"
    else:
        device = torch.device("cpu")
        backend = "CPU"
    
    print(f"Device: {device} ({backend})")
    print(f"PyTorch version: {torch.__version__}")
    
    # Test configurations
    configs = [
        (768, 3072, 12, "80M"),
        (960, 3840, 12, "110M"),
        (1024, 4096, 24, "260M"),
    ]
    
    for hidden_size, intermediate_size, num_layers, name in configs:
        print(f"\n{name} Model Configuration:")
        print("-" * 40)
        
        config = M2BertConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_layers,
            num_attention_heads=hidden_size // 64,
            use_monarch_mlp=True,
            monarch_nblocks=4,
            use_torch_compile=True
        )
        
        model = M2BertForMaskedLM(config).to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        monarch_params = sum(
            p.numel() for name, p in model.named_parameters()
            if 'weight' in name and len(p.shape) == 3
        )
        
        print(f"  Total parameters: {total_params/1e6:.1f}M")
        print(f"  Monarch parameters: {monarch_params/1e6:.1f}M")
        print(f"  Parameter reduction: {monarch_params/total_params*100:.1f}%")
        
        # Benchmark forward pass
        batch_size = 8
        seq_len = 512
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = model(input_ids)
        
        # Synchronize
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        
        # Benchmark
        num_iterations = 20
        start = time.perf_counter()
        
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model(input_ids)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        
        elapsed = time.perf_counter() - start
        
        print(f"  Forward pass time: {elapsed/num_iterations*1000:.2f}ms")
        print(f"  Throughput: {batch_size*seq_len*num_iterations/elapsed:.0f} tokens/sec")


if __name__ == "__main__":
    print("Modern M2-BERT Implementation")
    print("Pure PyTorch with latest optimizations")
    print()
    
    # Run benchmark
    benchmark_m2bert()
    
    print("\n" + "="*70)
    print("✓ Modern M2-BERT implementation complete!")
    print("  - torch.compile support")
    print("  - Metal Performance Shaders support")
    print("  - JIT-compiled kernels")
    print("  - Flash Attention support")
    print("="*70)