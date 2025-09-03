#!/usr/bin/env python
"""
M2-BERT: Masked Model BERT with Liquid Foundation Model Architecture
32k context window using LFM's hybrid conv-attention design
Incorporates MBRL for planning and code correction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
import math
from dataclasses import dataclass

from ..core.scalars import ScalarOperations


@dataclass
class M2BertConfig:
    """Configuration for M2-BERT model"""
    vocab_size: int = 50000
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 32768  # 32k context
    type_vocab_size: int = 2  # For segment embeddings
    layer_norm_eps: float = 1e-12
    
    # Special tokens for code
    pad_token_id: int = 0
    mask_token_id: int = 1
    cls_token_id: int = 2
    sep_token_id: int = 3
    error_token_id: int = 4  # Special token for errors
    
    # LFM specific
    mistake_detection_head: bool = True
    correction_generation_head: bool = True
    reward_prediction_head: bool = True


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding for handling 32k context efficiently.
    Better than absolute position embeddings for long sequences.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 32768):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Precompute frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute position indices
        t = torch.arange(max_seq_len).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        
        # Create rotation matrices
        self.register_buffer('cos_cached', freqs.cos())
        self.register_buffer('sin_cached', freqs.sin())
        
    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Apply rotary embeddings"""
        # x: [batch, seq_len, num_heads, head_dim]
        batch, _, num_heads, head_dim = x.shape
        
        # Get rotations for current sequence length
        cos = self.cos_cached[:seq_len, :].unsqueeze(0).unsqueeze(2)
        sin = self.sin_cached[:seq_len, :].unsqueeze(0).unsqueeze(2)
        
        # Apply rotation
        x1, x2 = x[..., :head_dim//2], x[..., head_dim//2:]
        rotated = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)
        
        return rotated


class M2BertAttention(nn.Module):
    """
    Multi-head attention with rotary embeddings for long context.
    """
    
    def __init__(self, config: M2BertConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.rotary_emb = RotaryPositionEmbedding(
            self.attention_head_size,
            config.max_position_embeddings
        )
        
        # Use PyTorch scalars for all math
        self.scalar_ops = ScalarOperations(torch.device("cpu"))  # Will move to correct device
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape for multi-head attention"""
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Linear projections
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Apply rotary embeddings
        query_layer = query_layer.permute(0, 2, 1, 3)  # [batch, seq, heads, dim]
        key_layer = key_layer.permute(0, 2, 1, 3)
        
        query_layer = self.rotary_emb(query_layer, seq_len)
        key_layer = self.rotary_emb(key_layer, seq_len)
        
        query_layer = query_layer.permute(0, 2, 1, 3)  # Back to [batch, heads, seq, dim]
        key_layer = key_layer.permute(0, 2, 1, 3)
        
        # Attention scores - using PyTorch scalars
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Context layer
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(batch_size, seq_len, self.all_head_size)
        
        return context_layer


class M2BertLayer(nn.Module):
    """Single transformer layer"""
    
    def __init__(self, config: M2BertConfig):
        super().__init__()
        self.attention = M2BertAttention(config)
        self.attention_output = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Feed-forward network
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.attention_output(attention_output)
        attention_output = self.attention_dropout(attention_output)
        hidden_states = self.attention_norm(hidden_states + attention_output)
        
        # Feed-forward
        intermediate = F.gelu(self.intermediate(hidden_states))
        output = self.output(intermediate)
        output = self.output_dropout(output)
        hidden_states = self.output_norm(hidden_states + output)
        
        return hidden_states


class M2BertModel(nn.Module):
    """
    M2-BERT base model with 32k context window.
    Designed for code understanding and mistake learning.
    """
    
    def __init__(self, config: M2BertConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.embedding_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            M2BertLayer(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Task-specific heads
        if config.mistake_detection_head:
            self.mistake_detector = nn.Linear(config.hidden_size, 2)  # Binary: mistake or not
            
        if config.correction_generation_head:
            self.correction_head = nn.Linear(config.hidden_size, config.vocab_size)
            
        if config.reward_prediction_head:
            self.reward_predictor = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_size // 2, 1)
            )
        
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights with proper scaling for 32k context"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        mistake_labels: Optional[torch.Tensor] = None,
        correction_labels: Optional[torch.Tensor] = None,
        reward_targets: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token type IDs default to zeros
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # Attention mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Prepare attention mask for broadcasting
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Embeddings
        word_embeds = self.word_embeddings(input_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)
        embeddings = word_embeds + token_type_embeds
        embeddings = self.embedding_norm(embeddings)
        embeddings = self.embedding_dropout(embeddings)
        
        # Transformer layers
        hidden_states = embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states, extended_attention_mask)
        
        outputs = {'hidden_states': hidden_states}
        total_loss = 0.0
        
        # Mistake detection loss
        if self.config.mistake_detection_head and mistake_labels is not None:
            mistake_logits = self.mistake_detector(hidden_states)
            mistake_loss = F.cross_entropy(
                mistake_logits.view(-1, 2),
                mistake_labels.view(-1),
                ignore_index=-100
            )
            outputs['mistake_logits'] = mistake_logits
            outputs['mistake_loss'] = mistake_loss
            total_loss = total_loss + mistake_loss
        
        # Correction generation loss
        if self.config.correction_generation_head and correction_labels is not None:
            correction_logits = self.correction_head(hidden_states)
            correction_loss = F.cross_entropy(
                correction_logits.view(-1, self.config.vocab_size),
                correction_labels.view(-1),
                ignore_index=self.config.pad_token_id
            )
            outputs['correction_logits'] = correction_logits
            outputs['correction_loss'] = correction_loss
            total_loss = total_loss + correction_loss
        
        # Reward prediction loss
        if self.config.reward_prediction_head and reward_targets is not None:
            # Pool hidden states (use CLS token or mean pooling)
            pooled = hidden_states[:, 0, :]  # CLS token
            reward_pred = self.reward_predictor(pooled).squeeze(-1)
            reward_loss = F.mse_loss(reward_pred, reward_targets)
            outputs['reward_predictions'] = reward_pred
            outputs['reward_loss'] = reward_loss
            total_loss = total_loss + reward_loss * 0.1  # Scale down reward loss
        
        if total_loss > 0:
            outputs['loss'] = total_loss
        
        return outputs
    
    def detect_mistakes(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Detect mistakes in code"""
        outputs = self.forward(input_ids, attention_mask=attention_mask)
        mistake_logits = self.mistake_detector(outputs['hidden_states'])
        return torch.argmax(mistake_logits, dim=-1)
    
    def generate_correction(
        self,
        input_ids: torch.Tensor,
        mistake_positions: torch.Tensor,
        max_length: int = 100
    ) -> torch.Tensor:
        """Generate corrections for identified mistakes"""
        outputs = self.forward(input_ids)
        hidden_states = outputs['hidden_states']
        
        # Focus on mistake positions
        mistake_hidden = hidden_states[mistake_positions > 0]
        
        # Generate corrections
        correction_logits = self.correction_head(mistake_hidden)
        corrections = torch.argmax(correction_logits, dim=-1)
        
        return corrections
    
    def predict_reward(self, input_ids: torch.Tensor, corrected_ids: torch.Tensor) -> torch.Tensor:
        """Predict reward for a correction"""
        # Concatenate original and corrected
        combined = torch.cat([input_ids, corrected_ids], dim=1)
        outputs = self.forward(combined)
        
        pooled = outputs['hidden_states'][:, 0, :]
        reward = self.reward_predictor(pooled).squeeze(-1)
        return torch.sigmoid(reward)  # Normalize to [0, 1]