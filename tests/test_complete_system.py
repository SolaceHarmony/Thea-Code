#!/usr/bin/env python
"""
Comprehensive pytest suite for Thea Code System
For the masochists who want to test EVERYTHING
"""

import pytest
import torch
import ray
import asyncio
import numpy as np
from typing import Any, Dict, List
import tempfile
import os

# Import all components we're testing
from thea_code_system.core import (
    BaseActor,
    EnhancedActor,
    EnhancedActorPool,
    TensorStore,
    ScalarOperations,
    ActorConfig
)

from thea_code_system.models.m2_bert_enhanced import (
    M2BertEnhanced,
    M2BertEnhancedConfig,
    LFM2StyleGLU,
    LFM2StyleConvBlock,
    LFM2StyleAttention
)

from thea_code_system.training.dpo_actor_trainer import (
    DPOTrainingActor,
    DPOConfig,
    DPOLoss
)


# ==================== Fixtures ====================

@pytest.fixture(scope="session")
def ray_context():
    """Initialize Ray for testing"""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=2)
    yield
    ray.shutdown()


@pytest.fixture
def device():
    """Get best available device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


@pytest.fixture
def scalar_ops(device):
    """Create scalar operations instance"""
    return ScalarOperations(device)


@pytest.fixture
def model_config():
    """Create test model configuration"""
    return M2BertEnhancedConfig(
        hidden_size=256,  # Small for testing
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=1000,
        max_position_embeddings=512
    )


@pytest.fixture
def sample_batch(device):
    """Create sample batch for testing"""
    batch_size = 2
    seq_len = 32
    return {
        'input_ids': torch.randint(0, 1000, (batch_size, seq_len), device=device),
        'attention_mask': torch.ones(batch_size, seq_len, device=device),
        'labels': torch.randint(0, 1000, (batch_size, seq_len), device=device)
    }


# ==================== Core Tests ====================

class TestScalarOperations:
    """Test that ALL math uses PyTorch tensors"""
    
    def test_scalar_creation(self, scalar_ops):
        """Test scalar tensor creation"""
        result = scalar_ops.scalar(42)
        assert isinstance(result, torch.Tensor)
        assert result.item() == 42
        assert result.device == scalar_ops.device
    
    def test_addition(self, scalar_ops):
        """Test addition uses PyTorch"""
        result = scalar_ops.add(1, 1)
        assert isinstance(result, torch.Tensor)
        assert result.item() == 2
    
    def test_multiplication(self, scalar_ops):
        """Test multiplication uses PyTorch"""
        result = scalar_ops.mul(3, 7)
        assert isinstance(result, torch.Tensor)
        assert result.item() == 21
    
    def test_division(self, scalar_ops):
        """Test division uses PyTorch"""
        result = scalar_ops.div(10, 2)
        assert isinstance(result, torch.Tensor)
        assert result.item() == 5
    
    def test_power(self, scalar_ops):
        """Test power uses PyTorch"""
        result = scalar_ops.pow(2, 3)
        assert isinstance(result, torch.Tensor)
        assert result.item() == 8
    
    def test_chain_operations(self, scalar_ops):
        """Test chaining operations maintains PyTorch tensors"""
        # (2 + 3) * 4 / 2 = 10
        result = scalar_ops.add(2, 3)
        result = scalar_ops.mul(result, 4)
        result = scalar_ops.div(result, 2)
        assert isinstance(result, torch.Tensor)
        assert result.item() == 10
    
    @pytest.mark.parametrize("value", [0, -1, 1.5, 1e10, -1e10])
    def test_edge_cases(self, scalar_ops, value):
        """Test edge cases remain as tensors"""
        result = scalar_ops.scalar(value)
        assert isinstance(result, torch.Tensor)
        assert result.item() == pytest.approx(value)


class TestTensorStore:
    """Test tensor sharing via Ray object store"""
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, ray_context):
        """Test basic store and retrieve"""
        store = TensorStore()
        
        # Create tensor
        tensor = torch.randn(10, 10)
        
        # Store
        ref = store.put("test_tensor", tensor)
        assert ref is not None
        
        # Retrieve
        retrieved = store.get("test_tensor")
        assert retrieved is not None
        assert torch.allclose(retrieved, tensor)
    
    @pytest.mark.asyncio
    async def test_delete(self, ray_context):
        """Test tensor deletion"""
        store = TensorStore()
        
        tensor = torch.randn(5, 5)
        store.put("to_delete", tensor)
        
        # Delete
        deleted = store.delete("to_delete")
        assert deleted == True
        
        # Should be gone
        retrieved = store.get("to_delete")
        assert retrieved is None
    
    @pytest.mark.asyncio
    async def test_list_keys(self, ray_context):
        """Test listing stored keys"""
        store = TensorStore()
        
        # Store multiple
        store.put("tensor1", torch.randn(3, 3))
        store.put("tensor2", torch.randn(3, 3))
        
        keys = store.list_keys()
        assert "tensor1" in keys
        assert "tensor2" in keys
    
    @pytest.mark.asyncio
    async def test_large_tensor(self, ray_context):
        """Test with large tensors"""
        store = TensorStore()
        
        # 100MB tensor
        large_tensor = torch.randn(5000, 5000)
        
        store.put("large", large_tensor)
        retrieved = store.get("large")
        
        assert torch.allclose(retrieved, large_tensor)


# ==================== Actor Tests ====================

class TestActorSystem:
    """Test distributed actor system"""
    
    @pytest.mark.asyncio
    async def test_enhanced_actor_creation(self, ray_context):
        """Test creating enhanced actors"""
        
        class TestActor(EnhancedActor):
            async def process(self, x):
                # Must use PyTorch
                tensor = self.scalar_ops.scalar(x)
                return self.scalar_ops.mul(tensor, 2).item()
        
        config = ActorConfig(name="test", num_cpus=1.0)
        pool = EnhancedActorPool(TestActor, num_actors=2, config=config)
        await pool.initialize()
        
        assert pool.num_actors == 2
        assert pool._initialized == True
        
        await pool.shutdown()
    
    @pytest.mark.asyncio
    async def test_actor_processing(self, ray_context):
        """Test actors process data correctly"""
        
        class ComputeActor(EnhancedActor):
            async def process(self, x):
                tensor = self.scalar_ops.scalar(x)
                result = self.scalar_ops.add(self.scalar_ops.mul(tensor, 2), 1)
                return result.item()
        
        config = ActorConfig(name="compute", num_cpus=1.0)
        pool = EnhancedActorPool(ComputeActor, num_actors=2, config=config)
        await pool.initialize()
        
        # Process data
        data = [1, 2, 3, 4, 5]
        results = pool.map(lambda actor, x: actor.process.remote(x), data)
        
        expected = [x * 2 + 1 for x in data]
        assert results == expected
        
        await pool.shutdown()
    
    @pytest.mark.asyncio
    async def test_actor_metrics(self, ray_context):
        """Test actor metrics tracking"""
        
        class MetricsActor(EnhancedActor):
            async def process(self, x):
                return x * 2
        
        config = ActorConfig(name="metrics", num_cpus=1.0)
        pool = EnhancedActorPool(MetricsActor, num_actors=1, config=config)
        await pool.initialize()
        
        # Process some data
        data = list(range(10))
        pool.map(lambda actor, x: actor.process.remote(x), data)
        
        # Check metrics
        assert pool.metrics.total_tasks_submitted == 10
        assert pool.metrics.total_tasks_completed == 10
        
        await pool.shutdown()


# ==================== Model Tests ====================

class TestM2BertEnhanced:
    """Test M2-BERT Enhanced model"""
    
    def test_model_creation(self, model_config):
        """Test model instantiation"""
        model = M2BertEnhanced(model_config)
        
        assert model is not None
        assert model.config == model_config
    
    def test_forward_pass(self, model_config, sample_batch):
        """Test forward pass"""
        model = M2BertEnhanced(model_config)
        model.eval()
        
        with torch.no_grad():
            outputs = model(**sample_batch)
        
        assert 'logits' in outputs
        assert 'loss' in outputs
        assert outputs['logits'].shape == (*sample_batch['input_ids'].shape, model_config.vocab_size)
    
    def test_lora_target_modules(self, model_config):
        """Test LoRA target modules match LFM2"""
        model = M2BertEnhanced(model_config)
        
        target_modules = model.get_lora_target_modules()
        
        # Should have GLU, MHA, and Conv modules
        assert 'w1' in target_modules
        assert 'w2' in target_modules
        assert 'w3' in target_modules
        assert 'q_proj' in target_modules
        assert 'k_proj' in target_modules
        assert 'v_proj' in target_modules
        assert 'out_proj' in target_modules
        assert 'in_proj' in target_modules
    
    def test_block_pattern(self, model_config):
        """Test hybrid block pattern"""
        # Check that pattern is generated correctly
        pattern = model_config.get_block_pattern()
        
        assert len(pattern) == model_config.num_hidden_layers
        assert 'conv' in pattern
        assert 'attention' in pattern
        
        # Should have more conv early, more attention late
        conv_early = pattern[:len(pattern)//2].count('conv')
        conv_late = pattern[len(pattern)//2:].count('conv')
        assert conv_early >= conv_late
    
    def test_glu_module(self, model_config):
        """Test GLU module"""
        glu = LFM2StyleGLU(model_config)
        
        x = torch.randn(2, 10, model_config.hidden_size)
        output = glu(x)
        
        assert output.shape == x.shape
        
        # Check that w1, w2, w3 exist
        assert hasattr(glu, 'w1')
        assert hasattr(glu, 'w2')
        assert hasattr(glu, 'w3')
    
    def test_conv_block(self, model_config):
        """Test convolution block"""
        conv_block = LFM2StyleConvBlock(model_config)
        
        x = torch.randn(2, 10, model_config.hidden_size)
        output = conv_block(x)
        
        assert output.shape == x.shape
        
        # Check modules
        assert hasattr(conv_block, 'in_proj')
        assert hasattr(conv_block, 'out_proj')
        assert hasattr(conv_block, 'conv')
    
    def test_attention_block(self, model_config):
        """Test attention block with GQA"""
        attn_block = LFM2StyleAttention(model_config)
        
        x = torch.randn(2, 10, model_config.hidden_size)
        output = attn_block(x)
        
        assert output.shape == x.shape
        
        # Check projections
        assert hasattr(attn_block, 'q_proj')
        assert hasattr(attn_block, 'k_proj')
        assert hasattr(attn_block, 'v_proj')
        assert hasattr(attn_block, 'out_proj')
        
        # Check GQA
        assert attn_block.num_key_value_heads < attn_block.num_attention_heads


# ==================== Training Tests ====================

class TestDPOTraining:
    """Test DPO training components"""
    
    def test_dpo_loss(self):
        """Test DPO loss computation"""
        loss_fn = DPOLoss(beta=0.1)
        
        # Create dummy log probs
        batch_size = 4
        policy_chosen = torch.randn(batch_size)
        policy_rejected = torch.randn(batch_size)
        ref_chosen = torch.randn(batch_size)
        ref_rejected = torch.randn(batch_size)
        
        loss = loss_fn(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert not torch.isnan(loss)
    
    @pytest.mark.asyncio
    async def test_dpo_actor_initialization(self, ray_context, model_config):
        """Test DPO training actor initialization"""
        actor_config = ActorConfig(name="dpo_test", num_cpus=1.0)
        
        # Create actor (local, not remote for testing)
        actor = DPOTrainingActor("test_actor", actor_config)
        
        # Mock tokenizer
        class MockTokenizer:
            pad_token_id = 0
        
        dpo_config = DPOConfig(
            use_lora=False,  # Disable for testing
            learning_rate=1e-6
        )
        
        await actor.initialize_dpo(
            M2BertEnhanced,
            model_config,
            MockTokenizer(),
            dpo_config
        )
        
        assert actor._initialized == True
        assert actor.model is not None
        assert actor.reference_model is not None
        assert actor.dpo_loss is not None
    
    def test_dpo_config(self):
        """Test DPO configuration"""
        config = DPOConfig()
        
        # Check LFM2 defaults
        assert config.learning_rate == 1e-6
        assert config.lora_r == 8
        assert config.lora_alpha == 16
        assert config.lora_dropout == 0.1
        assert config.beta == 0.1


# ==================== Integration Tests ====================

class TestIntegration:
    """End-to-end integration tests"""
    
    @pytest.mark.asyncio
    async def test_full_pipeline(self, ray_context, model_config):
        """Test complete pipeline from actors to model"""
        
        # Create model
        model = M2BertEnhanced(model_config)
        
        # Create actor pool
        class ModelActor(EnhancedActor):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.model = None
            
            async def initialize(self):
                await super().initialize()
                self.model = M2BertEnhanced(model_config)
                self.model.eval()
            
            async def process(self, input_ids):
                with torch.no_grad():
                    outputs = self.model(input_ids)
                return outputs['logits'].shape[-1]  # Return vocab size
        
        config = ActorConfig(name="model", num_cpus=1.0)
        pool = EnhancedActorPool(ModelActor, num_actors=2, config=config)
        await pool.initialize()
        
        # Process through actors
        input_ids = torch.randint(0, 1000, (1, 10))
        results = pool.map(
            lambda actor, x: actor.process.remote(x),
            [input_ids] * 2
        )
        
        assert all(r == model_config.vocab_size for r in results)
        
        await pool.shutdown()
    
    @pytest.mark.asyncio
    async def test_pytorch_everywhere(self, ray_context):
        """Verify PyTorch is used for ALL operations"""
        
        class MathActor(EnhancedActor):
            async def process(self, x):
                # Chain of operations - all must use PyTorch
                result = self.scalar_ops.scalar(x)
                result = self.scalar_ops.add(result, 10)
                result = self.scalar_ops.mul(result, 2)
                result = self.scalar_ops.div(result, 4)
                return result.item()
        
        config = ActorConfig(name="math", num_cpus=1.0)
        pool = EnhancedActorPool(MathActor, num_actors=1, config=config)
        await pool.initialize()
        
        # Test: (5 + 10) * 2 / 4 = 7.5
        results = pool.map(lambda actor, x: actor.process.remote(x), [5])
        
        assert results[0] == 7.5
        
        await pool.shutdown()


# ==================== Performance Tests ====================

class TestPerformance:
    """Performance and stress tests"""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_throughput(self, ray_context):
        """Test system throughput"""
        
        class FastActor(EnhancedActor):
            async def process(self, x):
                return x * 2
        
        config = ActorConfig(name="perf", num_cpus=1.0)
        pool = EnhancedActorPool(FastActor, num_actors=4, config=config)
        await pool.initialize()
        
        # Process large batch
        data = list(range(1000))
        import time
        
        start = time.time()
        results = pool.map(lambda actor, x: actor.process.remote(x), data)
        elapsed = time.time() - start
        
        throughput = len(data) / elapsed
        
        # Should achieve reasonable throughput
        assert throughput > 100  # At least 100 items/sec
        assert len(results) == len(data)
        
        await pool.shutdown()
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_memory_efficiency(self, ray_context):
        """Test memory usage with tensor sharing"""
        store = TensorStore()
        
        # Store many tensors
        for i in range(100):
            tensor = torch.randn(100, 100)
            store.put(f"tensor_{i}", tensor)
        
        # Should handle this without issues
        assert len(store.list_keys()) == 100
        
        # Clear
        store.clear()
        assert len(store.list_keys()) == 0


# ==================== Run Configuration ====================

if __name__ == "__main__":
    # Run with: pytest tests/test_complete_system.py -v
    # For slow tests: pytest tests/test_complete_system.py -v -m slow
    # For specific class: pytest tests/test_complete_system.py::TestScalarOperations -v
    pytest.main([__file__, "-v"])