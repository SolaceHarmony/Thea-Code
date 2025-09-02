#!/usr/bin/env python
"""
Quick CLI verification of the Thea Code System
Run this to verify everything is ship-shape!
"""

import sys
import os
import torch
import ray
import asyncio
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

console = Console()


def check_imports():
    """Check all critical imports"""
    console.print("\n[bold cyan]🔍 Checking Imports...[/bold cyan]")
    
    imports_status = []
    
    # Core imports
    try:
        from thea_code_system.core import (
            BaseActor, EnhancedActorPool, TensorStore, ScalarOperations
        )
        imports_status.append(("Core Components", "✅"))
    except ImportError as e:
        imports_status.append(("Core Components", f"❌ {str(e)}"))
    
    # Model imports
    try:
        from thea_code_system.models.m2_bert_enhanced import (
            M2BertEnhanced, M2BertEnhancedConfig
        )
        imports_status.append(("M2-BERT Enhanced", "✅"))
    except ImportError as e:
        imports_status.append(("M2-BERT Enhanced", f"❌ {str(e)}"))
    
    # Training imports
    try:
        from thea_code_system.training.dpo_actor_trainer import (
            DPOTrainingActor, ActorBasedDPOTrainer, DPOConfig
        )
        imports_status.append(("DPO Training", "✅"))
    except ImportError as e:
        imports_status.append(("DPO Training", f"❌ {str(e)}"))
    
    # External dependencies
    try:
        import transformers
        imports_status.append((f"Transformers {transformers.__version__}", "✅"))
    except:
        imports_status.append(("Transformers", "❌ Not installed"))
    
    try:
        import peft
        imports_status.append((f"PEFT {peft.__version__}", "✅"))
    except:
        imports_status.append(("PEFT", "❌ Not installed"))
    
    try:
        import trl
        imports_status.append((f"TRL {trl.__version__}", "✅"))
    except:
        imports_status.append(("TRL", "❌ Not installed"))
    
    # Display results
    table = Table(title="Import Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    
    for component, status in imports_status:
        table.add_row(component, status)
    
    console.print(table)
    
    return all("✅" in status for _, status in imports_status)


def check_device_setup():
    """Check PyTorch device configuration"""
    console.print("\n[bold cyan]🖥️  Checking Device Setup...[/bold cyan]")
    
    device_info = []
    
    # CUDA
    if torch.cuda.is_available():
        device_info.append(("CUDA", f"✅ {torch.cuda.get_device_name(0)}"))
        device_info.append(("CUDA Memory", f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"))
    else:
        device_info.append(("CUDA", "❌ Not available"))
    
    # MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        device_info.append(("MPS (Apple Silicon)", "✅ Available"))
    else:
        device_info.append(("MPS", "Not available"))
    
    # CPU
    device_info.append(("CPU", f"✅ {torch.get_num_threads()} threads"))
    
    # Display
    table = Table(title="Device Configuration")
    table.add_column("Device", style="cyan")
    table.add_column("Status", style="green")
    
    for device, status in device_info:
        table.add_row(device, status)
    
    console.print(table)
    
    return True


async def test_scalar_operations():
    """Test that ALL math uses PyTorch"""
    console.print("\n[bold cyan]🔢 Testing PyTorch Scalar Operations...[/bold cyan]")
    
    from thea_code_system.core import ScalarOperations
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    scalar_ops = ScalarOperations(device)
    
    # Test basic operations
    tests = []
    
    # Addition
    result = scalar_ops.add(1, 1)
    tests.append(("1 + 1", isinstance(result, torch.Tensor) and result.item() == 2))
    
    # Multiplication
    result = scalar_ops.mul(3, 7)
    tests.append(("3 * 7", isinstance(result, torch.Tensor) and result.item() == 21))
    
    # Division
    result = scalar_ops.div(10, 2)
    tests.append(("10 / 2", isinstance(result, torch.Tensor) and result.item() == 5))
    
    # Display
    table = Table(title="PyTorch Scalar Operations")
    table.add_column("Operation", style="cyan")
    table.add_column("Status", style="green")
    
    for op, passed in tests:
        status = "✅ Uses PyTorch" if passed else "❌ Failed"
        table.add_row(op, status)
    
    console.print(table)
    
    return all(passed for _, passed in tests)


async def test_actor_system():
    """Test Ray actor system"""
    console.print("\n[bold cyan]🎭 Testing Actor System...[/bold cyan]")
    
    # Initialize Ray if needed with custom temp dir on large volume
    if not ray.is_initialized():
        # Use the 4TB volume for Ray temp storage
        import os
        temp_dir = "/Volumes/emberstuff/ray_temp"
        os.makedirs(temp_dir, exist_ok=True)
        os.environ["RAY_object_spilling_directory"] = os.path.join(temp_dir, "spill")
        
        ray.init(
            ignore_reinit_error=True, 
            num_cpus=2,
            _temp_dir=temp_dir,
            object_store_memory=500_000_000,  # 500MB for testing
            logging_level="warning"  # Less verbose
        )
    
    tests = []
    
    try:
        from thea_code_system.core import EnhancedActor, EnhancedActorPool, ActorConfig
        
        # Create simple test actor
        class TestActor(EnhancedActor):
            async def process(self, x):
                return x * 2
        
        # Create pool
        config = ActorConfig(name="test", num_cpus=1.0)
        pool = EnhancedActorPool(TestActor, num_actors=2, config=config)
        await pool.initialize()
        
        # Test processing
        data = [1, 2, 3, 4]
        results = pool.map(lambda actor, x: actor.process.remote(x), data)
        
        tests.append(("Actor Pool Creation", True))
        tests.append(("Distributed Processing", results == [2, 4, 6, 8]))
        tests.append(("Pool Size", pool.num_actors == 2))
        
        # Cleanup
        await pool.shutdown()
        
    except Exception as e:
        tests.append(("Actor System", f"❌ {str(e)}"))
    
    # Display
    table = Table(title="Actor System Status")
    table.add_column("Test", style="cyan")
    table.add_column("Result", style="green")
    
    for test, result in tests:
        status = "✅ Passed" if result == True else (result if isinstance(result, str) else "❌ Failed")
        table.add_row(test, status)
    
    console.print(table)
    
    return all(r == True for _, r in tests)


async def test_model_creation():
    """Test M2-BERT Enhanced model creation"""
    console.print("\n[bold cyan]🤖 Testing M2-BERT Enhanced...[/bold cyan]")
    
    tests = []
    
    try:
        from thea_code_system.models.m2_bert_enhanced import (
            M2BertEnhanced, M2BertEnhancedConfig
        )
        
        # Create config
        config = M2BertEnhancedConfig(
            hidden_size=768,
            num_hidden_layers=6,  # Small for testing
            vocab_size=1000  # Small for testing
        )
        
        # Create model
        model = M2BertEnhanced(config)
        tests.append(("Model Creation", True))
        
        # Check parameter count
        param_count = sum(p.numel() for p in model.parameters())
        tests.append(("Parameter Count", f"{param_count:,}"))
        
        # Check LFM2 modules
        lora_modules = model.get_lora_target_modules()
        expected_modules = ['w1', 'w2', 'w3', 'q_proj', 'k_proj', 'v_proj', 'out_proj', 'in_proj', 'out_proj']
        tests.append(("LoRA Target Modules", lora_modules == expected_modules))
        
        # Test forward pass
        input_ids = torch.randint(0, 1000, (1, 10))
        outputs = model(input_ids)
        tests.append(("Forward Pass", 'logits' in outputs))
        
    except Exception as e:
        tests.append(("M2-BERT Enhanced", f"❌ {str(e)}"))
    
    # Display
    table = Table(title="M2-BERT Enhanced Status")
    table.add_column("Test", style="cyan")
    table.add_column("Result", style="green")
    
    for test, result in tests:
        if isinstance(result, bool):
            status = "✅ Passed" if result else "❌ Failed"
        else:
            status = str(result)
        table.add_row(test, status)
    
    console.print(table)
    
    return all(r == True for _, r in tests if isinstance(r, bool))


async def main():
    """Run all verification tests"""
    console.print(Panel.fit(
        "[bold cyan]🚀 Thea Code System Verification[/bold cyan]\n"
        "Checking that everything is ship-shape!",
        border_style="cyan"
    ))
    
    all_passed = True
    
    # Run checks
    all_passed &= check_imports()
    all_passed &= check_device_setup()
    all_passed &= await test_scalar_operations()
    all_passed &= await test_actor_system()
    all_passed &= await test_model_creation()
    
    # Summary
    if all_passed:
        console.print(Panel.fit(
            "[bold green]✅ All Systems Operational![/bold green]\n"
            "The Thea Code System is ready for deployment!",
            border_style="green"
        ))
    else:
        console.print(Panel.fit(
            "[bold red]⚠️ Some Issues Detected[/bold red]\n"
            "Please check the failed components above.",
            border_style="red"
        ))
    
    # Cleanup Ray
    if ray.is_initialized():
        ray.shutdown()
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)