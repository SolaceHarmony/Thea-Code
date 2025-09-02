#!/usr/bin/env python
"""
M2-BERT with Liquid Foundation Model Architecture
Combines M2-BERT's 32k context with LFM's hybrid conv-attention blocks
Uses MBRL for planning-based code correction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
import math
from dataclasses import dataclass

from ..core.scalars import ScalarOperations


@dataclass
class M2BertLFMConfig:
    """Configuration for M2-BERT with LFM architecture"""
    # Model dimensions
    vocab_size: int = 50000
    hidden_size: int = 768
    intermediate_size: int = 3072
    max_position_embeddings: int = 32768  # 32k context
    
    # LFM hybrid architecture
    n_conv_blocks: int = 10  # Short convolution blocks
    n_attention_blocks: int = 6  # Attention blocks
    conv_kernel_size: int = 3  # Short conv as per LFM
    
    # Attention configuration
    num_attention_heads: int = 12
    num_kv_heads: int = 4  # For grouped-query attention
    attention_dropout: float = 0.1
    
    # Multiplicative gating (Poli's innovation)
    use_multiplicative_gates: bool = True
    gate_activation: str = "silu"  # SwiGLU-style gating
    
    # Training configuration
    hidden_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12
    
    # Special tokens
    pad_token_id: int = 0
    mask_token_id: int = 1
    cls_token_id: int = 2
    sep_token_id: int = 3
    
    # MBRL planning
    use_planning_head: bool = True
    planning_horizon: int = 10
    
    # Hardware optimization
    target_device: str = "auto"
    optimize_for_latency: bool = True
    
    def get_block_pattern(self) -> List[str]:
        """
        Get STAR-optimized block pattern.
        Real STAR would optimize this, but we use a good heuristic:
        More conv blocks early (local patterns), attention later (global)
        """
        # Pattern from LFM paper: conv-heavy early, attention-heavy late
        pattern = (
            ['conv'] * 3 +      # Local feature extraction
            ['attn'] +          # First global view
            ['conv'] * 4 +      # More local processing
            ['attn'] * 2 +      # Global reasoning
            ['conv'] * 3 +      # Refinement
            ['attn'] * 3        # Final global understanding
        )
        return pattern[:self.n_conv_blocks + self.n_attention_blocks]


class MultiplicativeGate(nn.Module):
    """
    Multiplicative gating mechanism from LFM.
    Key to efficient sequence modeling.
    """
    
    def __init__(self, dim: int, activation: str = "silu"):
        super().__init__()
        self.dim = dim
        self.activation = activation
        
        # Projects to (value, gate_B, gate_C)
        self.proj = nn.Linear(dim, dim * 3)
        
        if activation == "silu":
            self.act = nn.SiLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply double multiplicative gating"""
        # Project to get value and two gates
        projected = self.proj(x)
        value, gate_b, gate_c = projected.chunk(3, dim=-1)
        
        # First gate (input-dependent)
        value = self.act(gate_b) * value
        
        # Second gate (output modulation)
        output = gate_c * value
        
        return output


class LFMConvBlock(nn.Module):
    """
    Short convolution block with multiplicative gating.
    Efficient for local pattern extraction in code.
    """
    
    def __init__(self, config: M2BertLFMConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Layer norm
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Multiplicative gating
        if config.use_multiplicative_gates:
            self.gate = MultiplicativeGate(config.hidden_size, config.gate_activation)
        else:
            self.gate = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Short convolution (key for efficiency)
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.conv_kernel_size,
            padding=config.conv_kernel_size // 2,
            groups=1  # Could use groups for more efficiency
        )
        
        # Output projection
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Process sequence with short convolution.
        hidden_states: [batch, seq_len, hidden_size]
        """
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        
        # Apply gating
        hidden_states = self.gate(hidden_states)
        
        # Convolution (need to transpose for Conv1d)
        hidden_states = hidden_states.transpose(1, 2)  # [batch, hidden, seq]
        hidden_states = self.conv(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)  # [batch, seq, hidden]
        
        # Output projection
        hidden_states = self.output(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Residual connection
        return residual + hidden_states


class LFMAttentionBlock(nn.Module):
    """
    Efficient attention block with grouped-query attention.
    Handles global dependencies in code.
    """
    
    def __init__(self, config: M2BertLFMConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # Grouped-query attention projections
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, self.head_dim * config.num_kv_heads)
        self.v_proj = nn.Linear(config.hidden_size, self.head_dim * config.num_kv_heads)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Normalization and dropout
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.attention_dropout)
        
        # For rotary embeddings (handle 32k context)
        self._init_rope()
        
    def _init_rope(self):
        """Initialize rotary position embeddings for long context"""
        dim = self.head_dim
        max_seq = 32768
        
        freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq).float()
        freqs = torch.outer(t, freqs).float()
        
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())
    
    def apply_rope(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Apply rotary position embeddings"""
        # x: [batch, seq, heads, head_dim]
        cos = self.cos_cached[:seq_len, :].unsqueeze(0).unsqueeze(2)
        sin = self.sin_cached[:seq_len, :].unsqueeze(0).unsqueeze(2)
        
        x1, x2 = x[..., :self.head_dim//2], x[..., self.head_dim//2:]
        rotated = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)
        
        return rotated
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply grouped-query attention.
        hidden_states: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        
        # Projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Apply rotary embeddings
        q = self.apply_rope(q, seq_len)
        k = self.apply_rope(k, seq_len)
        
        # Grouped-query attention (repeat k,v for each query group)
        if self.num_kv_heads < self.num_heads:
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
        
        # Transpose for attention
        q = q.transpose(1, 2)  # [batch, heads, seq, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            scores = scores + attention_mask
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        output = self.o_proj(attn_output)
        output = self.dropout(output)
        
        return residual + output


class MBRLPlanningHead(nn.Module):
    """
    MBRL planning head for code correction.
    Predicts future states and rewards for planning.
    """
    
    def __init__(self, config: M2BertLFMConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.planning_horizon = config.planning_horizon
        
        # State transition model
        self.transition = nn.LSTM(
            config.hidden_size,
            config.hidden_size,
            num_layers=2,
            batch_first=True
        )
        
        # Reward prediction
        self.reward_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, 1)
        )
        
        # Value estimation
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, 1)
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        plan_steps: int = 5
    ) -> Dict[str, torch.Tensor]:
        """
        Plan future corrections.
        Returns predicted rewards and values for planning.
        """
        batch_size = hidden_states.size(0)
        
        # Use CLS token for planning
        state = hidden_states[:, 0, :].unsqueeze(1)  # [batch, 1, hidden]
        
        planned_states = []
        rewards = []
        values = []
        
        # Roll out future states
        for _ in range(plan_steps):
            # Predict next state
            state, _ = self.transition(state)
            planned_states.append(state)
            
            # Predict reward and value
            reward = self.reward_head(state.squeeze(1))
            value = self.value_head(state.squeeze(1))
            
            rewards.append(reward)
            values.append(value)
        
        return {
            'planned_states': torch.stack(planned_states, dim=1),
            'predicted_rewards': torch.stack(rewards, dim=1),
            'predicted_values': torch.stack(values, dim=1)
        }


class M2BertLFM(nn.Module):
    """
    M2-BERT with Liquid Foundation Model architecture.
    Combines efficient conv blocks with strategic attention.
    """
    
    def __init__(self, config: M2BertLFMConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.embedding_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.embedding_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Build hybrid blocks according to pattern
        self.blocks = nn.ModuleList()
        block_pattern = config.get_block_pattern()
        
        for block_type in block_pattern:
            if block_type == 'conv':
                self.blocks.append(LFMConvBlock(config))
            elif block_type == 'attn':
                self.blocks.append(LFMAttentionBlock(config))
        
        # Task heads
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        
        # MBRL planning head
        if config.use_planning_head:
            self.planning_head = MBRLPlanningHead(config)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights appropriately"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_planning: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through M2-BERT-LFM.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Target labels for training
            use_planning: Whether to use MBRL planning
        
        Returns:
            Dictionary with logits, loss, and optionally planning outputs
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create position IDs
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        word_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        hidden_states = word_embeds + position_embeds
        hidden_states = self.embedding_norm(hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)
        
        # Process through hybrid blocks
        for block in self.blocks:
            if isinstance(block, LFMAttentionBlock) and attention_mask is not None:
                # Prepare attention mask
                extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_mask = (1.0 - extended_mask) * -10000.0
                hidden_states = block(hidden_states, extended_mask)
            else:
                hidden_states = block(hidden_states)
        
        # Language modeling head
        lm_logits = self.lm_head(hidden_states)
        
        outputs = {'logits': lm_logits, 'hidden_states': hidden_states}
        
        # Calculate loss if labels provided
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
            outputs['loss'] = loss
        
        # MBRL planning if requested
        if use_planning and self.config.use_planning_head:
            planning_outputs = self.planning_head(hidden_states)
            outputs.update(planning_outputs)
        
        return outputs
    
    def generate_correction(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        use_planning: bool = True
    ) -> torch.Tensor:
        """
        Generate code corrections using the model.
        Can use MBRL planning for better corrections.
        """
        self.eval()
        device = input_ids.device
        batch_size = input_ids.size(0)
        
        # Get initial hidden states
        outputs = self.forward(input_ids, use_planning=use_planning)
        
        if use_planning and 'predicted_rewards' in outputs:
            # Use planning to guide generation
            # Select actions with highest predicted rewards
            rewards = outputs['predicted_rewards']
            best_actions = torch.argmax(rewards, dim=1)
        
        # Simple greedy generation for now
        generated = input_ids
        for _ in range(max_length):
            outputs = self.forward(generated)
            next_token_logits = outputs['logits'][:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if all sequences have generated SEP token
            if (next_token == self.config.sep_token_id).all():
                break
        
        return generated