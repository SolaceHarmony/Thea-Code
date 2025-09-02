#!/usr/bin/env python
"""
M2-BERT Hybrid: Our Ray-powered, PyTorch-everywhere version of LFM2's architecture
Taking the best of LFM2's hybrid conv/attention pattern and making it distributed!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import ray
from typing import Dict, Any, Optional, List, Tuple
import math

from ..core.scalars import ScalarOperations

# Import Monarch matrix operations
try:
    import sys
    import os
    # Add project root to path
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    sys.path.insert(0, project_root)
    
    from blockdiag_linear import BlockdiagLinear
    from blockdiag_butterfly_multiply import blockdiag_butterfly_multiply
    MONARCH_AVAILABLE = True
    print("✅ Monarch matrices loaded successfully!")
except ImportError as e:
    print(f"Warning: Monarch matrices not available ({e}), using standard Linear layers")
    BlockdiagLinear = nn.Linear
    MONARCH_AVAILABLE = False


@ray.remote(max_concurrency=2)
class HybridLayerActor:
    """
    Ray actor for a single hybrid layer
    Can be either conv or attention based on configuration
    """
    
    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        layer_type: str,  # "conv" or "attention"
        device: str = "mps"
    ):
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.layer_type = layer_type
        self.device = torch.device(device if torch.backends.mps.is_available() else "cpu")
        
        # Use PyTorch for EVERYTHING
        self.scalar_ops = ScalarOperations(self.device)
        
        # Initialize the appropriate layer type
        if layer_type == "conv":
            self._init_conv_layer()
        else:
            self._init_attention_layer()
        
        # Shared MLP (GLU style like LFM2)
        self._init_mlp()
        
    def _init_conv_layer(self):
        """Initialize convolutional sequence mixer using REAL causal conv1d"""
        from ..ops.causal_conv1d import CausalConv1d
        
        # Use our proper causal conv implementation
        self.conv = CausalConv1d(
            channels=self.hidden_size,
            kernel_size=3,  # Short conv like LFM2  
            bias=False,
            activation=None  # We'll apply activation separately
        ).to(self.device)
        
        # Input and output projections (like LFM2's in_proj, out_proj)
        self.in_proj = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=False).to(self.device)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False).to(self.device)
        
    def _init_attention_layer(self):
        """Initialize attention layer (WITH position encoding)"""
        # Standard multi-head attention
        self.num_heads = 8  # Simplified for now
        self.head_dim = self.hidden_size // self.num_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False).to(self.device)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False).to(self.device)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False).to(self.device)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False).to(self.device)
        
        # Simple RoPE for attention layers ONLY
        self._init_rope()
        
    def _init_rope(self):
        """Initialize RoPE for attention layers"""
        max_seq_len = 2048  # Start smaller
        dim = self.head_dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        self.cos_cached = freqs.cos().to(self.device)
        self.sin_cached = freqs.sin().to(self.device)
        
    def _init_mlp(self):
        """Initialize GLU MLP with Monarch matrices (M2-BERT style)"""
        intermediate_size = self.hidden_size * 4
        
        # Use Monarch matrices for MLP if available (M2-BERT innovation!)
        if MONARCH_AVAILABLE:
            print(f"  Layer {self.layer_idx}: Using Monarch matrices for MLP")
            self.w1 = BlockdiagLinear(
                self.hidden_size, 
                intermediate_size, 
                bias=False,
                nblocks=4  # M2-BERT default
            ).to(self.device)
            self.w2 = BlockdiagLinear(
                intermediate_size, 
                self.hidden_size, 
                bias=False,
                nblocks=4
            ).to(self.device)
            self.w3 = BlockdiagLinear(
                self.hidden_size, 
                intermediate_size, 
                bias=False,
                nblocks=4
            ).to(self.device)
        else:
            # Fallback to standard linear layers
            self.w1 = nn.Linear(self.hidden_size, intermediate_size, bias=False).to(self.device)
            self.w2 = nn.Linear(intermediate_size, self.hidden_size, bias=False).to(self.device)
            self.w3 = nn.Linear(self.hidden_size, intermediate_size, bias=False).to(self.device)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(self.hidden_size).to(self.device)
        self.norm2 = nn.LayerNorm(self.hidden_size).to(self.device)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - sync method for ThreadedActor
        """
        # Move to device
        hidden_states = hidden_states.to(self.device)
        residual = hidden_states
        
        # Apply the appropriate layer type
        if self.layer_type == "conv":
            hidden_states = self._conv_forward(self.norm1(hidden_states))
        else:
            hidden_states = self._attention_forward(self.norm1(hidden_states))
        
        hidden_states = hidden_states + residual
        
        # GLU MLP
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.w2(F.silu(self.w1(hidden_states)) * self.w3(hidden_states))
        hidden_states = hidden_states + residual
        
        return hidden_states.cpu()  # Move back for Ray serialization
    
    def _conv_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convolutional forward using REAL causal conv (no position encoding!)"""
        batch_size, seq_len, _ = x.shape
        
        # Project to B, C, x components (like LFM2's gated structure)
        BCx = self.in_proj(x).transpose(-1, -2)  # [batch, 3*hidden, seq]
        B, C, x_conv = BCx.chunk(3, dim=1)
        
        # Apply gate
        Bx = B * x_conv
        
        # Apply REAL causal convolution (no trimming needed - it's handled internally!)
        conv_out = self.conv(Bx)  # This is proper causal now
        
        # Apply second gate and project back
        y = C * conv_out
        y = self.out_proj(y.transpose(-1, -2))
        
        return y
    
    def _attention_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attention forward (WITH position encoding)"""
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE to Q and K (attention layers need position!)
        q = self._apply_rope(q, seq_len)
        k = self._apply_rope(k, seq_len)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(self.device)
        scores.masked_fill_(mask, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.out_proj(attn_output)
        
        return output
    
    def _apply_rope(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Apply RoPE (for attention layers only)"""
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        
        rx1 = x1 * cos - x2 * sin
        rx2 = x1 * sin + x2 * cos
        
        return torch.stack([rx1, rx2], dim=-1).flatten(-2)


class M2BertHybrid(nn.Module):
    """
    Our hybrid M2-BERT with Ray-distributed layers
    Borrowing LFM2's architecture but making it OURS
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 768,
        num_layers: int = 12,
        layer_types: Optional[List[str]] = None,
        use_ray: bool = True
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_ray = use_ray
        
        # Decide layer types (borrowing LFM2's pattern)
        if layer_types is None:
            # Default pattern: conv-heavy early, attention-heavy late
            # This is the SECRET SAUCE from LFM2!
            layer_types = []
            for i in range(num_layers):
                # Use PyTorch to decide! 
                threshold = torch.tensor(i / num_layers)
                use_attention = torch.rand(1) < threshold  # More likely attention as we go deeper
                layer_types.append("attention" if use_attention.item() else "conv")
        
        self.layer_types = layer_types
        
        # Embeddings
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.embed_norm = nn.LayerNorm(hidden_size)
        
        # Create layers (as Ray actors if enabled)
        if use_ray:
            self._init_ray_layers()
        else:
            self._init_local_layers()
        
        # Output layers for BERT tasks
        self.final_norm = nn.LayerNorm(hidden_size)
        
        # MLM head with Monarch matrices (M2-BERT style)
        if MONARCH_AVAILABLE:
            print("🎯 Using Monarch matrices for MLM head")
            self.mlm_head = BlockdiagLinear(hidden_size, vocab_size, bias=False, nblocks=4)
        else:
            self.mlm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Keep standard LM head for compatibility
        self.lm_head = self.mlm_head
        
    def _init_ray_layers(self):
        """Initialize layers as Ray actors"""
        if not ray.is_initialized():
            ray.init(
                num_cpus=4,
                _temp_dir="/Volumes/emberstuff/ray_temp",
                object_store_memory=1_000_000_000
            )
        
        self.layer_actors = []
        for i, layer_type in enumerate(self.layer_types):
            actor = HybridLayerActor.remote(
                layer_idx=i,
                hidden_size=self.hidden_size,
                layer_type=layer_type,
                device="mps"
            )
            self.layer_actors.append(actor)
            
        print(f"Created {len(self.layer_actors)} Ray actors")
        print(f"Layer types: {self.layer_types}")
        
    def _init_local_layers(self):
        """Initialize layers locally (fallback)"""
        # TODO: Implement local version
        raise NotImplementedError("Local layers not yet implemented")
    
    def forward(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through our hybrid architecture
        """
        # Embed
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.embed_norm(hidden_states)
        
        # Process through hybrid layers
        if self.use_ray:
            # Process with Ray actors
            for i, actor in enumerate(self.layer_actors):
                # Ray remote call (async but we wait)
                future = actor.forward.remote(hidden_states)
                hidden_states = ray.get(future)
                hidden_states = torch.tensor(hidden_states)  # Ensure it's a tensor
        else:
            # Local processing
            for layer in self.layers:
                hidden_states = layer(hidden_states)
        
        # Output
        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return {"logits": logits}
    
    def forward_mlm(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        BERT-style MLM forward pass
        """
        # Standard forward through hybrid layers
        outputs = self.forward(input_ids)
        logits = outputs["logits"]
        
        results = {"logits": logits}
        
        # Compute MLM loss if labels provided
        if labels is not None:
            # Only compute loss on masked tokens (labels != -100)
            loss_fct = nn.CrossEntropyLoss()
            
            # Flatten for loss computation
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(-1, vocab_size)
            labels_flat = labels.view(-1)
            
            # Mask out non-masked tokens (MLM only trains on [MASK] positions)
            mask = labels_flat != -100
            if mask.any():
                masked_logits = logits_flat[mask]
                masked_labels = labels_flat[mask]
                mlm_loss = loss_fct(masked_logits, masked_labels)
            else:
                mlm_loss = torch.tensor(0.0, device=logits.device)
            
            results["loss"] = mlm_loss
            results["mlm_loss"] = mlm_loss
        
        return results
    
    def get_layer_distribution(self) -> Dict[str, int]:
        """Analyze the distribution of layer types"""
        conv_count = sum(1 for t in self.layer_types if t == "conv")
        attn_count = sum(1 for t in self.layer_types if t == "attention")
        
        return {
            "conv_layers": conv_count,
            "attention_layers": attn_count,
            "pattern": self.layer_types
        }


# Test it!
if __name__ == "__main__":
    print("🚀 Testing M2-BERT Hybrid with Ray actors...")
    
    # Create model with specific pattern
    layer_pattern = ["conv", "conv", "conv", "attention", "conv", "attention", "attention", "attention"]
    
    model = M2BertHybrid(
        hidden_size=256,  # Smaller for testing
        num_layers=8,
        layer_types=layer_pattern,
        use_ray=True
    )
    
    print(f"\n📊 Layer distribution: {model.get_layer_distribution()}")
    
    # Test forward pass
    input_ids = torch.randint(0, 1000, (1, 10))
    print(f"\n🔥 Running forward pass with Ray actors...")
    
    outputs = model(input_ids)
    print(f"✅ Output shape: {outputs['logits'].shape}")
    
    print("\n💫 Our hybrid architecture is alive!")
    print("Conv layers: No position encoding (implicit in convolution)")
    print("Attention layers: RoPE for position")
    print("All layers: GLU MLP with w1, w2, w3")
    print("Everything: PyTorch tensors and Ray actors!")