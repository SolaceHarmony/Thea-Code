#!/usr/bin/env python
"""
Test Training Infrastructure
Validates actor-based training with proper async flow
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import ray
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset
import time

from thea_code_system.core.training import (
    TrainingActor,
    TrainingConfig,
    ParameterServerActor,
    DataLoaderActor,
    DistributedTrainer
)
from thea_code_system.core.base import ActorConfig


# Simple test model
class SimpleModel(nn.Module):
    """Simple feedforward network for testing"""
    
    def __init__(self, input_size: int = 10, hidden_size: int = 20, output_size: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, inputs=None, labels=None, **kwargs):
        """Forward pass with loss calculation"""
        # Handle both dict and tensor inputs
        if inputs is None and 'input_ids' in kwargs:
            inputs = kwargs['input_ids']
            
        x = self.fc1(inputs)
        x = self.relu(x)
        logits = self.fc2(x)
        
        outputs = {'logits': logits}
        
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            outputs['loss'] = loss
            
        return outputs


def create_dummy_dataset(num_samples: int = 1000, input_size: int = 10):
    """Create dummy dataset for testing"""
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, 2, (num_samples,))
    return TensorDataset(X, y)


async def test_single_actor_training():
    """Test training with a single actor"""
    print("\n📊 Test 1: Single Actor Training")
    print("-" * 40)
    
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    try:
        # Create training actor
        config = ActorConfig(name="test_trainer", num_cpus=1.0)
        TrainerClass = ray.remote(TrainingActor)
        actor = TrainerClass.remote("trainer_0", config)
        
        # Initialize training
        training_config = TrainingConfig(
            batch_size=32,
            learning_rate=0.001,
            gradient_accumulation_steps=1
        )
        
        await actor.initialize_training.remote(
            model_class=SimpleModel,
            model_kwargs={'input_size': 10, 'hidden_size': 20, 'output_size': 2},
            optimizer_class=optim.Adam,
            optimizer_kwargs={'lr': training_config.learning_rate},
            training_config=training_config
        )
        
        print("✅ Training actor initialized")
        
        # Create dummy batch
        batch = {
            'inputs': torch.randn(32, 10),
            'labels': torch.randint(0, 2, (32,))
        }
        
        # Train for a few steps
        losses = []
        for i in range(5):
            result = await actor.train_batch.remote(batch)
            losses.append(result['loss'])
            print(f"   Step {i+1}: Loss={result['loss']:.4f}, GradNorm={result['grad_norm']:.4f}")
        
        # Check that loss is changing (model is learning)
        if len(set(losses)) > 1:  # Losses are different
            print("✅ Model is training (loss changing)")
        else:
            print("❌ Model not training properly")
            return False
        
        # Get model state
        state = await actor.get_model_state.remote()
        if 'model_state_dict' in state and 'optimizer_state_dict' in state:
            print("✅ Model state can be retrieved")
        else:
            print("❌ Model state retrieval failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Single actor training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_parameter_server():
    """Test parameter server for weight synchronization"""
    print("\n📊 Test 2: Parameter Server")
    print("-" * 40)
    
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    try:
        # Create parameter server
        ps = ParameterServerActor.remote()
        
        # Initialize with dummy model state
        model = SimpleModel()
        model_state = model.state_dict()
        
        ray.get(ps.initialize.remote(model_state, num_actors=2))
        print("✅ Parameter server initialized")
        
        # Test weight retrieval
        weights = ray.get(ps.get_weights.remote())
        if weights:
            print(f"✅ Retrieved weights: {len(weights)} parameters")
        else:
            print("❌ Weight retrieval failed")
            return False
        
        # Test gradient accumulation
        dummy_grads = [torch.randn_like(p) for p in model.parameters()]
        ray.get(ps.accumulate_gradients.remote(dummy_grads, "actor_0"))
        ray.get(ps.accumulate_gradients.remote(dummy_grads, "actor_1"))
        
        # Average gradients
        averaged = ray.get(ps.average_gradients.remote())
        if len(averaged) == len(dummy_grads):
            print(f"✅ Gradient averaging works: {len(averaged)} gradients")
        else:
            print("❌ Gradient averaging failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Parameter server test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_data_loader_actor():
    """Test data loader actor"""
    print("\n📊 Test 3: Data Loader Actor")
    print("-" * 40)
    
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    try:
        # Create data loader actor
        DataLoaderClass = ray.remote(DataLoaderActor)
        loader = DataLoaderClass.remote("data_loader", None)
        
        # Create dataset
        dataset = create_dummy_dataset(100, 10)
        
        # Initialize
        await loader.initialize_data.remote(
            dataset=dataset,
            batch_size=16,
            shuffle=True
        )
        
        print("✅ Data loader actor initialized")
        
        # Get batches
        batches = []
        for i in range(3):
            batch = await loader.get_batch.remote()
            if batch and 'inputs' in batch:
                batches.append(batch)
                print(f"   Batch {i+1}: Shape={batch['inputs'].shape}")
        
        if len(batches) == 3:
            print("✅ Data loading works")
        else:
            print("❌ Data loading failed")
            return False
        
        # Test prefetching
        prefetched = await loader.prefetch_batches.remote(5)
        if len(prefetched) == 5:
            print(f"✅ Prefetching works: {len(prefetched)} batches")
        else:
            print("❌ Prefetching failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Data loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_distributed_training():
    """Test full distributed training system"""
    print("\n📊 Test 4: Distributed Training")
    print("-" * 40)
    
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    try:
        # Create trainer
        training_config = TrainingConfig(
            batch_size=16,
            learning_rate=0.001,
            gradient_accumulation_steps=2,
            log_every=2
        )
        
        trainer = DistributedTrainer(
            num_training_actors=2,
            training_config=training_config
        )
        
        # Create dataset
        dataset = create_dummy_dataset(200, 10)
        
        # Initialize
        await trainer.initialize(
            model_class=SimpleModel,
            model_kwargs={'input_size': 10, 'hidden_size': 20, 'output_size': 2},
            optimizer_class=optim.Adam,
            optimizer_kwargs={'lr': training_config.learning_rate},
            dataset=dataset
        )
        
        print("✅ Distributed trainer initialized")
        
        # Train for a few steps
        print("\n🚀 Training for 10 steps...")
        metrics = await trainer.train_epoch(10)
        
        print(f"\n📊 Training Results:")
        print(f"   Average Loss: {metrics['total_loss']/metrics['steps']:.4f}")
        print(f"   Throughput: {metrics['throughput']:.1f} samples/sec")
        print(f"   Total Steps: {metrics['steps']}")
        
        if metrics['steps'] == 10:
            print("✅ Distributed training completed successfully")
        else:
            print("❌ Training didn't complete all steps")
            return False
        
        # Cleanup
        await trainer.shutdown()
        
        return True
        
    except Exception as e:
        print(f"❌ Distributed training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_gradient_synchronization():
    """Test gradient synchronization between actors"""
    print("\n📊 Test 5: Gradient Synchronization")
    print("-" * 40)
    
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    try:
        # Create two training actors
        actors = []
        for i in range(2):
            config = ActorConfig(name=f"sync_test_{i}", num_cpus=1.0)
            TrainerClass = ray.remote(TrainingActor)
            actor = TrainerClass.remote(f"sync_{i}", config)
            
            training_config = TrainingConfig(batch_size=16)
            
            await actor.initialize_training.remote(
                model_class=SimpleModel,
                model_kwargs={'input_size': 10},
                optimizer_class=optim.SGD,
                optimizer_kwargs={'lr': 0.01},
                training_config=training_config
            )
            
            actors.append(actor)
        
        print("✅ Created 2 actors for sync test")
        
        # Create different batches for each actor
        batch1 = {
            'inputs': torch.randn(16, 10),
            'labels': torch.randint(0, 2, (16,))
        }
        batch2 = {
            'inputs': torch.randn(16, 10),
            'labels': torch.randint(0, 2, (16,))
        }
        
        # Forward and backward on both actors
        result1 = await actors[0].forward_pass.remote(batch1)
        result2 = await actors[1].forward_pass.remote(batch2)
        
        # Get loss values
        loss1, outputs1 = ray.get(result1)
        loss2, outputs2 = ray.get(result2)
        
        # Now do backward pass with the loss tensors
        grad_norm1 = await actors[0].backward_pass.remote(loss1)
        grad_norm2 = await actors[1].backward_pass.remote(loss2)
        
        print(f"   Actor 0 loss: {loss1.item():.4f}")
        print(f"   Actor 1 loss: {loss2.item():.4f}")
        
        # Get gradients from both actors
        grads1 = await actors[0].get_gradients.remote()
        grads2 = await actors[1].get_gradients.remote()
        
        # Average gradients manually
        averaged_grads = []
        for g1, g2 in zip(grads1, grads2):
            averaged_grads.append((g1 + g2) / 2)
        
        # Sync averaged gradients to both actors
        await actors[0].sync_gradients.remote(averaged_grads)
        await actors[1].sync_gradients.remote(averaged_grads)
        
        # Verify both actors have same gradients now
        final_grads1 = await actors[0].get_gradients.remote()
        final_grads2 = await actors[1].get_gradients.remote()
        
        all_match = True
        for g1, g2 in zip(final_grads1, final_grads2):
            if not torch.allclose(g1, g2, rtol=1e-5):
                all_match = False
                break
        
        if all_match:
            print("✅ Gradient synchronization successful")
        else:
            print("❌ Gradients don't match after sync")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Gradient sync test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all training tests"""
    print("\n" + "="*60)
    print("🎯 TRAINING INFRASTRUCTURE TEST")
    print("="*60)
    
    results = {}
    
    # Test 1: Single actor
    results['Single Actor'] = await test_single_actor_training()
    
    # Test 2: Parameter server
    results['Parameter Server'] = await test_parameter_server()
    
    # Test 3: Data loader
    results['Data Loader'] = await test_data_loader_actor()
    
    # Test 4: Distributed training
    results['Distributed'] = await test_distributed_training()
    
    # Test 5: Gradient sync
    results['Gradient Sync'] = await test_gradient_synchronization()
    
    # Summary
    print("\n" + "="*60)
    print("📊 TRAINING TEST SUMMARY")
    print("="*60)
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name:20s}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 Training infrastructure validated!")
        print("Key capabilities demonstrated:")
        print("  ✅ Async training flow with actors")
        print("  ✅ Gradient accumulation and synchronization")
        print("  ✅ Parameter server for weight management")
        print("  ✅ Distributed data loading")
        print("  ✅ Multi-actor coordinated training")
    else:
        print("\n⚠️ Some training tests failed")
    
    # Cleanup
    if ray.is_initialized():
        ray.shutdown()
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)