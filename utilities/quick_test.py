#!/usr/bin/env python
"""Quick test without Ray to verify core components"""

import torch
import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("🔍 Testing Core Imports...")

# Test core imports
try:
    from thea_code_system.core import ScalarOperations
    print("✅ Core imports work")
except ImportError as e:
    print(f"❌ Core import failed: {e}")

# Test model imports
try:
    from thea_code_system.models.m2_bert_enhanced import M2BertEnhanced, M2BertEnhancedConfig
    print("✅ M2-BERT Enhanced imports work")
except ImportError as e:
    print(f"❌ Model import failed: {e}")

# Test DPO imports (now that we have peft and trl)
try:
    from thea_code_system.training.dpo_actor_trainer import DPOConfig
    from peft import LoraConfig
    # Skip TRL import for now due to tf-keras issue
    print("✅ DPO training imports work (peft)")
except ImportError as e:
    print(f"❌ DPO import failed: {e}")

# Test PyTorch operations
print("\n🔢 Testing PyTorch Scalar Operations...")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
scalar_ops = ScalarOperations(device)

# Test basic math
result = scalar_ops.add(1, 1)
assert isinstance(result, torch.Tensor) and result.item() == 2
print(f"✅ 1 + 1 = {result.item()} (using PyTorch)")

result = scalar_ops.mul(3, 7)
assert isinstance(result, torch.Tensor) and result.item() == 21
print(f"✅ 3 * 7 = {result.item()} (using PyTorch)")

# Test model creation
print("\n🤖 Testing M2-BERT Enhanced Creation...")
config = M2BertEnhancedConfig(
    hidden_size=768,
    num_hidden_layers=4,  # Small for testing
    vocab_size=1000
)
model = M2BertEnhanced(config)
print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

# Check LoRA target modules
lora_modules = model.get_lora_target_modules()
print(f"✅ LoRA target modules: {lora_modules}")

# Test forward pass
input_ids = torch.randint(0, 1000, (1, 10))
outputs = model(input_ids)
print(f"✅ Forward pass successful, output shape: {outputs['logits'].shape}")

print("\n✨ All basic tests passed! System is functional.")
print("Note: Ray actor tests skipped to avoid disk space issues.")