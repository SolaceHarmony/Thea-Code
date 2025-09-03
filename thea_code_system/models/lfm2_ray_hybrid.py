#!/usr/bin/env python
"""
LFM2 Ray Hybrid: Our PyTorch+Ray version of Liquid AI's LFM2 architecture
Copied from transformers/models/lfm2 and adapted for Ray distribution + M2-BERT Monarch matrices

Original copyright: HuggingFace Team 2025 - Apache 2.0 License
Our adaptations: PyTorch everywhere + Ray actors + Monarch matrices
"""

import torch
import torch.nn.functional as F
from torch import nn
import ray
import math
from typing import Any, Optional, Union, Dict, List, Tuple
from dataclasses import dataclass

# Import Monarch matrix operations (our existing setup)
try:
    import sys
    import os
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    sys.path.insert(0, project_root)
    
    from blockdiag_linear import BlockdiagLinear
    from blockdiag_butterfly_multiply import blockdiag_butterfly_multiply
    MONARCH_AVAILABLE = True
    print("✅ Monarch matrices loaded for LFM2 hybrid!")
except ImportError as e:
    print(f"Warning: Monarch matrices not available ({e})")
    BlockdiagLinear = nn.Linear
    MONARCH_AVAILABLE = False

# Import our causal conv (we have this working)
# Add parent directories to path for imports
thea_code_path = os.path.join(project_root, 'thea_code_system')
sys.path.insert(0, thea_code_path)

from ops.causal_conv1d import CausalConv1d, causal_conv1d_fn
from core.scalars import ScalarOperations

@dataclass
class LFM2RayConfig:
    """Configuration for our LFM2 Ray hybrid"""
    vocab_size: int = 32000
    hidden_size: int = 2048
    num_hidden_layers: int = 16
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    intermediate_size: int = 12288
    max_position_embeddings: int = 128000
    rope_theta: float = 1000000.0
    norm_eps: float = 1e-5
    use_cache: bool = True
    
    # LFM2 specific
    block_auto_adjust_ff_dim: bool = True
    block_ff_dim: int = 12288
    block_ffn_dim_multiplier: float = 1.0
    block_multiple_of: int = 256
    block_use_swiglu: bool = True
    
    # Conv specific
    conv_L_cache: int = 3
    conv_bias: bool = False
    conv_dim: int = 2048
    conv_dim_out: int = 2048
    
    # Layer pattern (conv vs attention)
    layer_types: List[str] = None  # Will default to LFM2 pattern
    full_attn_idxs: List[int] = None  # Explicit attention indices
    
    # Our additions
    use_ray: bool = True
    use_monarch_mlp: bool = True
    monarch_mlp_nblocks: int = 4
    device: str = "mps"

class LFM2RayRMSNorm(nn.Module):
    """RMSNorm with PyTorch scalars and Ray compatibility"""
    
    def __init__(self, hidden_size, eps=1e-6, device="mps"):
        super().__init__()
        self.device = torch.device(device if torch.backends.mps.is_available() else "cpu")
        self.weight = nn.Parameter(torch.ones(hidden_size, device=self.device))
        self.variance_epsilon = eps
        self.scalar_ops = ScalarOperations(self.device)
        
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        
        # Use PyTorch scalars for variance computation
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(
            self.scalar_ops.add(variance, torch.tensor(self.variance_epsilon, device=self.device))
        )
        
        return self.weight * hidden_states.to(input_dtype)

class LFM2RayRotaryEmbedding(nn.Module):
    """Advanced RoPE with PyTorch scalars"""
    
    def __init__(self, config: LFM2RayConfig):
        super().__init__()
        self.device = torch.device(config.device if torch.backends.mps.is_available() else "cpu")
        self.scalar_ops = ScalarOperations(self.device)
        
        self.max_seq_len_cached = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        
        dim = config.hidden_size // config.num_attention_heads
        
        # Create inverse frequencies using PyTorch scalars
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq.to(self.device), persistent=False)
        
    def forward(self, x, position_ids):
        # Expand using PyTorch operations
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        
        # Compute frequencies
        freqs = torch.matmul(inv_freq_expanded, position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = emb.cos()
        sin = emb.sin()
        
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class LFM2RayMLP(nn.Module):
    """LFM2 MLP with Monarch matrices and PyTorch scalars"""
    
    def __init__(self, config: LFM2RayConfig):
        super().__init__()
        self.device = torch.device(config.device if torch.backends.mps.is_available() else "cpu")
        self.scalar_ops = ScalarOperations(self.device)
        
        # Auto-adjust intermediate size (LFM2 feature)
        intermediate_size = config.intermediate_size
        if config.block_auto_adjust_ff_dim:
            # LFM2's dimension adjustment logic
            intermediate_size = int(2 * intermediate_size / 3)
            if config.block_ffn_dim_multiplier is not None:
                intermediate_size = int(config.block_ffn_dim_multiplier * intermediate_size)
                intermediate_size = config.block_multiple_of * (
                    (intermediate_size + config.block_multiple_of - 1) // config.block_multiple_of
                )
        
        print(f"  MLP: {config.hidden_size} → {intermediate_size} → {config.hidden_size}")
        
        # Use Monarch matrices if available (M2-BERT enhancement)
        if config.use_monarch_mlp and MONARCH_AVAILABLE:
            print(f"  Using Monarch matrices with {config.monarch_mlp_nblocks} blocks")
            self.w1 = BlockdiagLinear(
                config.hidden_size, intermediate_size, 
                bias=False, nblocks=config.monarch_mlp_nblocks
            ).to(self.device)
            self.w2 = BlockdiagLinear(
                intermediate_size, config.hidden_size,
                bias=False, nblocks=config.monarch_mlp_nblocks  
            ).to(self.device)
            self.w3 = BlockdiagLinear(
                config.hidden_size, intermediate_size,
                bias=False, nblocks=config.monarch_mlp_nblocks
            ).to(self.device)
        else:
            # Standard linear layers
            self.w1 = nn.Linear(config.hidden_size, intermediate_size, bias=False).to(self.device)
            self.w2 = nn.Linear(intermediate_size, config.hidden_size, bias=False).to(self.device)
            self.w3 = nn.Linear(config.hidden_size, intermediate_size, bias=False).to(self.device)
        
    def forward(self, x):
        # SwiGLU activation (like LFM2)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

@ray.remote
class LFM2RayLayerActor:
    """
    Ray actor for a single LFM2 layer (conv or attention)
    This is where Ray distribution magic happens!
    """
    
    def __init__(
        self,
        config: LFM2RayConfig,
        layer_idx: int,
        layer_type: str  # "conv" or "full_attention"
    ):
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        self.device = torch.device(config.device if torch.backends.mps.is_available() else "cpu")
        self.scalar_ops = ScalarOperations(self.device)
        
        print(f"🚀 Layer {layer_idx}: {layer_type} actor starting...")
        
        # Initialize the appropriate layer type
        if layer_type == "full_attention":
            self._init_attention_layer()
        else:
            self._init_conv_layer()
            
        # Shared MLP and norms
        self.feed_forward = LFM2RayMLP(config)
        self.operator_norm = LFM2RayRMSNorm(config.hidden_size, config.norm_eps, config.device)
        self.ffn_norm = LFM2RayRMSNorm(config.hidden_size, config.norm_eps, config.device)
        
        print(f"✅ Layer {layer_idx} actor ready!")
        
    def _init_attention_layer(self):
        """Initialize attention with RoPE"""
        self.num_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.head_dim = self.config.hidden_size // self.num_heads
        
        # Q, K, V projections
        self.q_proj = nn.Linear(self.config.hidden_size, self.num_heads * self.head_dim, bias=False).to(self.device)
        self.k_proj = nn.Linear(self.config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False).to(self.device)
        self.v_proj = nn.Linear(self.config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False).to(self.device)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.config.hidden_size, bias=False).to(self.device)
        
        # RoPE 
        self.rotary_emb = LFM2RayRotaryEmbedding(self.config)
        
    def _init_conv_layer(self):
        """Initialize conv layer using our causal conv"""
        self.conv_dim = self.config.conv_dim
        self.conv_dim_out = self.config.conv_dim_out
        
        # Input and output projections (LFM2 style)
        self.in_proj = nn.Linear(self.config.hidden_size, 3 * self.conv_dim, bias=False).to(self.device)
        self.out_proj = nn.Linear(self.conv_dim_out, self.config.hidden_size, bias=False).to(self.device)
        
        # Our causal conv implementation
        self.conv = CausalConv1d(
            channels=self.conv_dim,
            kernel_size=3,  # LFM2 uses short kernels
            bias=self.config.conv_bias,
            activation=None
        ).to(self.device)
        
    def _apply_rope(self, q, k, position_ids):
        """Apply RoPE to Q and K"""
        cos, sin = self.rotary_emb(q, position_ids)
        
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        
        return q_embed, k_embed
        
    def _rotate_half(self, x):
        """Rotate half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through this layer"""
        
        # Move to device
        hidden_states = hidden_states.to(self.device)
        if position_ids is not None:
            position_ids = position_ids.to(self.device)
            
        # Residual connection
        residual = hidden_states
        
        # Apply operator norm (pre-norm)
        hidden_states = self.operator_norm(hidden_states)
        
        # Layer-specific processing
        if self.layer_type == "full_attention":
            hidden_states = self._attention_forward(hidden_states, position_ids, attention_mask)
        else:
            hidden_states = self._conv_forward(hidden_states)
            
        # First residual
        hidden_states = hidden_states + residual
        
        # FFN with residual
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = hidden_states + residual
        
        return hidden_states.cpu()  # Return to CPU for Ray serialization
        
    def _attention_forward(self, hidden_states, position_ids, attention_mask):
        """Attention forward with RoPE"""
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        if position_ids is not None:
            query_states, key_states = self._apply_rope(query_states, key_states, position_ids)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.config.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output
        
    def _conv_forward(self, hidden_states):
        """Conv forward using our causal conv"""
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to B, C, x (LFM2 gating structure)
        BCx = self.in_proj(hidden_states)
        B, C, x = BCx.chunk(3, dim=-1)
        
        # First gate: B * x
        Bx = B * x
        
        # Apply causal convolution
        Bx = Bx.transpose(-1, -2)  # [batch, channels, seq]
        conv_out = self.conv(Bx)  # Our causal conv
        
        # Second gate: C * conv_out
        y = C * conv_out.transpose(-1, -2)  # Back to [batch, seq, channels]
        
        # Output projection
        y = self.out_proj(y)
        
        return y

class LFM2RayHybrid(nn.Module):
    """
    Complete LFM2 + Ray + Monarch hybrid architecture
    """
    
    def __init__(self, config: LFM2RayConfig = None):
        super().__init__()
        
        if config is None:
            config = LFM2RayConfig()
        self.config = config
        
        print("🏗️ Building LFM2 Ray Hybrid...")
        
        # Set up layer types (LFM2 pattern)
        if config.layer_types is None:
            # Default LFM2 pattern: mostly conv, some attention
            config.layer_types = ["conv"] * config.num_hidden_layers
            if config.full_attn_idxs is None:
                # Default attention positions (from LFM2-1.2B config)
                config.full_attn_idxs = [2, 5, 8, 10, 12, 14]
            
            for idx in config.full_attn_idxs:
                if idx < config.num_hidden_layers:
                    config.layer_types[idx] = "full_attention"
        
        self.layer_types = config.layer_types
        
        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Initialize Ray if needed
        if config.use_ray and not ray.is_initialized():
            ray.init(
                num_cpus=4,
                _temp_dir="/Volumes/emberstuff/ray_temp",
                object_store_memory=1_000_000_000
            )
        
        # Create layer actors
        if config.use_ray:
            self._init_ray_layers()
        else:
            raise NotImplementedError("Local layers not implemented yet")
            
        # Output layers
        self.norm = LFM2RayRMSNorm(config.hidden_size, config.norm_eps, config.device)
        
        # MLM head with Monarch matrices
        if MONARCH_AVAILABLE:
            print("🎯 Using Monarch matrices for output head")
            self.lm_head = BlockdiagLinear(
                config.hidden_size, config.vocab_size, 
                bias=False, nblocks=4
            )
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        print(f"✅ LFM2 Ray Hybrid ready!")
        print(f"📊 Architecture: {self.get_layer_distribution()}")
        
    def _init_ray_layers(self):
        """Initialize layers as Ray actors"""
        self.layer_actors = []
        
        for i, layer_type in enumerate(self.layer_types):
            actor = LFM2RayLayerActor.remote(self.config, i, layer_type)
            self.layer_actors.append(actor)
            
        print(f"🚀 Created {len(self.layer_actors)} Ray layer actors")
        
    def forward(self, input_ids: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through hybrid architecture"""
        
        # Embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Generate position_ids if not provided
        if position_ids is None:
            batch_size, seq_len = input_ids.shape
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Create causal attention mask
        batch_size, seq_len = input_ids.shape
        attention_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)
        
        # Forward through Ray actors
        for i, actor in enumerate(self.layer_actors):
            future = actor.forward.remote(hidden_states, position_ids, attention_mask)
            hidden_states = ray.get(future)
            hidden_states = torch.tensor(hidden_states, requires_grad=True)
            
        # Final norm and output
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return {"logits": logits}
        
    def get_layer_distribution(self) -> Dict[str, Any]:
        """Analyze layer distribution"""
        conv_count = sum(1 for t in self.layer_types if t == "conv")
        attn_count = sum(1 for t in self.layer_types if t == "full_attention")
        
        return {
            "conv_layers": conv_count,
            "attention_layers": attn_count,
            "pattern": self.layer_types,
            "total_layers": len(self.layer_types)
        }

# Test it!
if __name__ == "__main__":
    print("🚀 Testing LFM2 Ray Hybrid Architecture!")
    
    # Create config
    config = LFM2RayConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=8,
        num_attention_heads=8,
        full_attn_idxs=[2, 4, 6],  # Some attention layers
        use_ray=True,
        use_monarch_mlp=True
    )
    
    # Create model
    model = LFM2RayHybrid(config)
    
    # Test forward
    input_ids = torch.randint(0, 1000, (2, 16))
    outputs = model(input_ids)
    
    print(f"\n✅ Success!")
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {outputs['logits'].shape}")
    print(f"Architecture: {model.get_layer_distribution()}")
    print(f"\n🎉 LFM2 + Ray + Monarch = Complete!")