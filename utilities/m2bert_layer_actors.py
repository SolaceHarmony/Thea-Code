#!/usr/bin/env python3
"""
M2-BERT Layer Actors
Each transformer layer as a Ray actor with out-of-band tensor communication
Enables pipeline parallelism and distributed layer computation
"""

import ray
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import asyncio
import time
from dataclasses import dataclass
import pickle
from m2bert_compatibility import MonarchLinearCompat, M2BertConfig


@dataclass
class TensorRef:
    """Reference to a tensor in shared memory"""
    shape: Tuple[int, ...]
    dtype: torch.dtype
    device: str
    data_ref: ray.ObjectRef


@ray.remote
class TensorStore:
    """Centralized tensor storage for out-of-band communication"""
    
    def __init__(self):
        self.tensors = {}
        self.tensor_id = 0
    
    def put(self, tensor: torch.Tensor) -> str:
        """Store tensor and return ID"""
        tensor_id = f"tensor_{self.tensor_id}"
        self.tensor_id += 1
        
        # Store tensor metadata and data separately
        self.tensors[tensor_id] = {
            'data': tensor.cpu().numpy(),  # Convert to numpy for serialization
            'shape': tensor.shape,
            'dtype': tensor.dtype,
            'device': str(tensor.device)
        }
        return tensor_id
    
    def get(self, tensor_id: str, target_device: Optional[str] = None) -> torch.Tensor:
        """Retrieve tensor by ID"""
        if tensor_id not in self.tensors:
            raise KeyError(f"Tensor {tensor_id} not found")
        
        tensor_info = self.tensors[tensor_id]
        tensor = torch.from_numpy(tensor_info['data']).to(tensor_info['dtype'])
        
        # Move to target device if specified
        if target_device:
            tensor = tensor.to(target_device)
        elif tensor_info['device'] != 'cpu':
            tensor = tensor.to(tensor_info['device'])
        
        return tensor
    
    def delete(self, tensor_id: str):
        """Remove tensor from storage"""
        if tensor_id in self.tensors:
            del self.tensors[tensor_id]


@ray.remote
class AttentionActor:
    """Actor for multi-head attention computation"""
    
    def __init__(self, config: M2BertConfig, layer_idx: int, tensor_store: ray.ObjectRef):
        self.config = config
        self.layer_idx = layer_idx
        self.tensor_store = tensor_store
        
        # Initialize attention weights
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        # Use Monarch linear if specified
        LinearClass = MonarchLinearCompat if config.use_monarch_mlp else nn.Linear
        
        self.query = LinearClass(self.hidden_size, self.hidden_size)
        self.key = LinearClass(self.hidden_size, self.hidden_size)
        self.value = LinearClass(self.hidden_size, self.hidden_size)
        self.output = LinearClass(self.hidden_size, self.hidden_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        # Device setup
        if torch.cuda.is_available():
            self.device = f"cuda:{layer_idx % torch.cuda.device_count()}"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        # Move weights to device
        self.query = self.query.to(self.device)
        self.key = self.key.to(self.device)
        self.value = self.value.to(self.device)
        self.output = self.output.to(self.device)
        
        print(f"AttentionActor {layer_idx} initialized on {self.device}")
    
    async def forward(self, hidden_states_id: str, attention_mask_id: Optional[str] = None) -> str:
        """Forward pass using tensor IDs"""
        
        # Retrieve tensors from store
        hidden_states = await self.tensor_store.get.remote(hidden_states_id, self.device)
        
        if attention_mask_id:
            attention_mask = await self.tensor_store.get.remote(attention_mask_id, self.device)
        else:
            attention_mask = None
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # Compute Q, K, V
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)
        
        # Reshape for multi-head attention
        query_layer = query_layer.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_layer = key_layer.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_layer = value_layer.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=True,
            enable_mem_efficient=True
        ) if torch.cuda.is_available() else torch.no_grad():
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_layer, key_layer, value_layer,
                attn_mask=attention_mask,
                dropout_p=self.dropout.p if self.training else 0.0
            )
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        attn_output = self.output(attn_output)
        attn_output = self.dropout(attn_output)
        
        # Store result and return ID
        output_id = await self.tensor_store.put.remote(attn_output)
        
        # Clean up input tensor if we're done with it
        await self.tensor_store.delete.remote(hidden_states_id)
        
        return output_id


@ray.remote
class MLPActor:
    """Actor for MLP/FFN computation"""
    
    def __init__(self, config: M2BertConfig, layer_idx: int, tensor_store: ray.ObjectRef):
        self.config = config
        self.layer_idx = layer_idx
        self.tensor_store = tensor_store
        
        # Initialize MLP weights
        LinearClass = MonarchLinearCompat if config.use_monarch_mlp else nn.Linear
        
        self.intermediate = LinearClass(config.hidden_size, config.intermediate_size)
        self.output = LinearClass(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Device setup
        if torch.cuda.is_available():
            self.device = f"cuda:{layer_idx % torch.cuda.device_count()}"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        self.intermediate = self.intermediate.to(self.device)
        self.output = self.output.to(self.device)
        
        print(f"MLPActor {layer_idx} initialized on {self.device}")
    
    async def forward(self, hidden_states_id: str) -> str:
        """Forward pass using tensor IDs"""
        
        # Retrieve tensor
        hidden_states = await self.tensor_store.get.remote(hidden_states_id, self.device)
        
        # MLP forward
        intermediate_output = torch.nn.functional.gelu(self.intermediate(hidden_states))
        output = self.output(intermediate_output)
        output = self.dropout(output)
        
        # Store result
        output_id = await self.tensor_store.put.remote(output)
        
        # Clean up
        await self.tensor_store.delete.remote(hidden_states_id)
        
        return output_id


@ray.remote
class LayerNormActor:
    """Actor for layer normalization"""
    
    def __init__(self, hidden_size: int, eps: float, layer_idx: int, tensor_store: ray.ObjectRef):
        self.layer_idx = layer_idx
        self.tensor_store = tensor_store
        
        self.norm = nn.LayerNorm(hidden_size, eps=eps)
        
        # Device setup
        if torch.cuda.is_available():
            self.device = f"cuda:{layer_idx % torch.cuda.device_count()}"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        self.norm = self.norm.to(self.device)
    
    async def forward(self, hidden_states_id: str, residual_id: str) -> str:
        """Apply layer norm with residual connection"""
        
        # Get tensors
        hidden_states = await self.tensor_store.get.remote(hidden_states_id, self.device)
        residual = await self.tensor_store.get.remote(residual_id, self.device)
        
        # Apply norm and residual
        output = self.norm(hidden_states + residual)
        
        # Store result
        output_id = await self.tensor_store.put.remote(output)
        
        # Clean up
        await self.tensor_store.delete.remote(hidden_states_id)
        
        return output_id


@ray.remote
class TransformerLayerActor:
    """Complete transformer layer as nested actors"""
    
    def __init__(self, config: M2BertConfig, layer_idx: int, tensor_store: ray.ObjectRef):
        self.config = config
        self.layer_idx = layer_idx
        self.tensor_store = tensor_store
        
        # Create sub-actors for this layer
        self.attention = AttentionActor.remote(config, layer_idx, tensor_store)
        self.mlp = MLPActor.remote(config, layer_idx, tensor_store)
        
        # Layer norms
        self.attention_norm = LayerNormActor.remote(
            config.hidden_size, config.layer_norm_eps, layer_idx, tensor_store
        )
        self.output_norm = LayerNormActor.remote(
            config.hidden_size, config.layer_norm_eps, layer_idx, tensor_store
        )
        
        print(f"TransformerLayerActor {layer_idx} initialized with nested actors")
    
    @ray.remote
    def forward_attention(self, hidden_states_id: str, attention_mask_id: Optional[str] = None) -> str:
        """Attention sub-layer with residual"""
        # Store residual
        residual_id = hidden_states_id
        
        # Attention
        attn_output_id = ray.get(self.attention.forward.remote(hidden_states_id, attention_mask_id))
        
        # Add residual and norm
        output_id = ray.get(self.attention_norm.forward.remote(attn_output_id, residual_id))
        
        return output_id
    
    @ray.remote
    def forward_mlp(self, hidden_states_id: str) -> str:
        """MLP sub-layer with residual"""
        # Store residual
        residual_id = hidden_states_id
        
        # MLP
        mlp_output_id = ray.get(self.mlp.forward.remote(hidden_states_id))
        
        # Add residual and norm
        output_id = ray.get(self.output_norm.forward.remote(mlp_output_id, residual_id))
        
        return output_id
    
    async def forward(self, hidden_states_id: str, attention_mask_id: Optional[str] = None) -> str:
        """Complete layer forward using nested remote functions"""
        
        # Chain attention and MLP using nested remote calls
        attn_output_id = await self.forward_attention.remote(hidden_states_id, attention_mask_id)
        output_id = await self.forward_mlp.remote(attn_output_id)
        
        return output_id


@ray.remote
class M2BertLayerPipeline:
    """Pipeline of transformer layers with out-of-band communication"""
    
    def __init__(self, config: M2BertConfig):
        self.config = config
        
        # Create tensor store
        self.tensor_store = TensorStore.remote()
        
        # Create layer actors
        self.layers = [
            TransformerLayerActor.remote(config, i, self.tensor_store)
            for i in range(config.num_hidden_layers)
        ]
        
        print(f"M2BertLayerPipeline initialized with {config.num_hidden_layers} layers")
    
    async def forward(self, input_tensor: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through all layers"""
        
        # Store initial tensors
        hidden_states_id = await self.tensor_store.put.remote(input_tensor)
        
        if attention_mask is not None:
            attention_mask_id = await self.tensor_store.put.remote(attention_mask)
        else:
            attention_mask_id = None
        
        # Pipeline through layers
        for i, layer in enumerate(self.layers):
            print(f"Processing layer {i}")
            hidden_states_id = await layer.forward.remote(hidden_states_id, attention_mask_id)
        
        # Retrieve final output
        output = await self.tensor_store.get.remote(hidden_states_id)
        
        return output
    
    async def forward_pipelined(self, batch: List[torch.Tensor]) -> List[torch.Tensor]:
        """Process multiple inputs in pipeline fashion"""
        
        # Start all inputs through the pipeline
        futures = []
        for input_tensor in batch:
            future = self.forward(input_tensor)
            futures.append(future)
            
            # Small delay to stagger pipeline stages
            await asyncio.sleep(0.01)
        
        # Collect results
        results = await asyncio.gather(*futures)
        return results


async def demo_layer_actors():
    """Demonstrate layer actor system"""
    print("="*70)
    print("M2-BERT LAYER ACTOR DEMO")
    print("="*70)
    
    ray.init(ignore_reinit_error=True)
    
    # Create config
    config = M2BertConfig(
        hidden_size=768,
        num_hidden_layers=4,  # Fewer layers for demo
        num_attention_heads=12,
        intermediate_size=3072,
        use_monarch_mlp=True,
        monarch_mlp_nblocks=4
    )
    
    # Create pipeline
    pipeline = M2BertLayerPipeline.remote(config)
    
    # Test input
    batch_size = 2
    seq_len = 128
    hidden_size = 768
    
    # Create test batch
    batch = [
        torch.randn(seq_len, hidden_size)
        for _ in range(batch_size)
    ]
    
    print(f"\nProcessing batch of {batch_size} sequences...")
    
    # Process batch
    start = time.perf_counter()
    results = await pipeline.forward_pipelined.remote(batch)
    elapsed = time.perf_counter() - start
    
    print(f"\nResults:")
    for i, result in enumerate(results):
        print(f"  Sequence {i}: shape={result.shape}, norm={result.norm().item():.4f}")
    
    print(f"\nTotal time: {elapsed:.3f}s")
    print(f"Time per sequence: {elapsed/batch_size:.3f}s")
    
    ray.shutdown()


if __name__ == "__main__":
    asyncio.run(demo_layer_actors())