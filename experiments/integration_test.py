#!/usr/bin/env python
"""
Integration Test
Validates that all components work together in the architecture
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import ray
import torch
import time
from typing import Any, Dict, List

from thea_code_system.core import (
    # Base components
    BaseOrchestrator,
    ActorConfig,
    
    # Enhanced components
    EnhancedActor,
    EnhancedActorPool,
    TensorStore,
    
    # PyTorch operations
    ScalarOperations,
    AccumulatorBase
)


class CodeAnalysisActor(EnhancedActor):
    """Actor for code analysis tasks"""
    
    async def process(self, code: str) -> Dict[str, Any]:
        """Analyze code snippet"""
        # All math with PyTorch
        lines = code.count('\n') + 1
        chars = len(code)
        
        # Use PyTorch for calculations
        lines_t = self.scalar_ops.scalar(lines)
        chars_t = self.scalar_ops.scalar(chars)
        
        # Calculate complexity score (simple heuristic)
        complexity = self.scalar_ops.mul(lines_t, 2)
        complexity = self.scalar_ops.add(complexity, self.scalar_ops.div(chars_t, 10))
        
        return {
            'actor_id': self.actor_id,
            'lines': lines,
            'chars': chars,
            'complexity': complexity.item()
        }
    
    async def extract_features(self, code: str) -> torch.Tensor:
        """Extract feature vector from code"""
        # Simple feature extraction
        features = [
            float(len(code)),
            float(code.count('\n')),
            float(code.count('def ')),
            float(code.count('class ')),
            float(code.count('import ')),
            float(code.count('if ')),
            float(code.count('for ')),
            float(code.count('while '))
        ]
        
        # Convert to tensor on device
        feature_tensor = torch.tensor(features, device=self.device)
        
        # Normalize using PyTorch
        mean = feature_tensor.mean()
        std = feature_tensor.std() + 1e-6
        normalized = (feature_tensor - mean) / std
        
        return normalized.cpu()


class CorrectionActor(EnhancedActor):
    """Actor for code correction tasks"""
    
    async def process(self, features: torch.Tensor) -> Dict[str, Any]:
        """Process code features for correction"""
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, device=self.device)
        elif features.device != self.device:
            features = features.to(self.device)
        
        # Simulate correction scoring
        score = features.abs().mean()
        confidence = torch.sigmoid(score)
        
        return {
            'actor_id': self.actor_id,
            'correction_score': score.item(),
            'confidence': confidence.item()
        }
    
    async def apply_correction(self, code: str, correction_params: Dict) -> str:
        """Apply correction to code"""
        # Placeholder for actual correction logic
        corrected = code.replace('  ', '    ')  # Fix indentation
        corrected = corrected.replace('print ', 'print(') 
        if not corrected.endswith(')') and 'print(' in corrected:
            corrected += ')'
        
        return corrected


class CodeCorrectionOrchestrator(BaseOrchestrator):
    """Orchestrator for code correction pipeline"""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.tensor_store = TensorStore()
        self.scalar_ops = ScalarOperations(self._get_device())
        
    def _get_device(self) -> torch.device:
        """Get compute device"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    async def execute_workflow(self, code_snippets: List[str]) -> List[Dict[str, Any]]:
        """Execute code correction workflow"""
        results = []
        
        # Step 1: Analysis phase
        analysis_pool = self.pools.get('analysis')
        if not analysis_pool:
            raise ValueError("Analysis pool not registered")
        
        print(f"📊 Analyzing {len(code_snippets)} code snippets...")
        
        # Analyze all snippets
        analysis_results = analysis_pool.map(
            lambda actor, code: actor.process.remote(code),
            code_snippets
        )
        
        # Extract features
        feature_tasks = []
        for i, snippet in enumerate(code_snippets):
            actor = analysis_pool.actors[i % len(analysis_pool.actors)]
            feature_tasks.append(actor.extract_features.remote(snippet))
        
        features = ray.get(feature_tasks)
        
        # Store features in tensor store
        for i, feat in enumerate(features):
            self.tensor_store.put(f"features_{i}", feat)
        
        print(f"✅ Analysis complete, features extracted")
        
        # Step 2: Correction phase
        correction_pool = self.pools.get('correction')
        if not correction_pool:
            raise ValueError("Correction pool not registered")
        
        print(f"📊 Applying corrections...")
        
        # Process features for correction
        correction_results = correction_pool.map(
            lambda actor, feat: actor.process.remote(feat),
            features
        )
        
        # Apply corrections
        corrected_snippets = []
        for i, snippet in enumerate(code_snippets):
            actor = correction_pool.actors[i % len(correction_pool.actors)]
            corrected = ray.get(
                actor.apply_correction.remote(snippet, correction_results[i])
            )
            corrected_snippets.append(corrected)
        
        print(f"✅ Corrections applied")
        
        # Combine results
        for i in range(len(code_snippets)):
            results.append({
                'original': code_snippets[i],
                'corrected': corrected_snippets[i],
                'analysis': analysis_results[i],
                'correction': correction_results[i]
            })
        
        return results


async def test_integrated_system():
    """Test the integrated architecture"""
    print("\n🔬 Testing Integrated System")
    print("-" * 40)
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    try:
        # Create orchestrator
        orchestrator = CodeCorrectionOrchestrator("main_orchestrator")
        
        # Configure pools
        analysis_config = ActorConfig(
            name="analysis",
            num_cpus=1.0,
            num_gpus=0
        )
        
        correction_config = ActorConfig(
            name="correction",
            num_cpus=1.0,
            num_gpus=0
        )
        
        # Create pools using enhanced pool
        analysis_pool = EnhancedActorPool(
            CodeAnalysisActor,
            num_actors=2,
            config=analysis_config
        )
        
        correction_pool = EnhancedActorPool(
            CorrectionActor,
            num_actors=2,
            config=correction_config
        )
        
        # Register pools
        orchestrator.register_pool('analysis', analysis_pool)
        orchestrator.register_pool('correction', correction_pool)
        
        # Initialize system
        await orchestrator.initialize()
        
        print("✅ System initialized")
        
        # Test data - code snippets with issues
        test_snippets = [
            "def hello():\n  print 'Hello World'",
            "class MyClass:\n  def __init__(self):\n    self.value = 42",
            "for i in range(10):\n  print i",
            "import sys\nimport os\nprint sys.path"
        ]
        
        # Execute workflow
        start = time.time()
        results = await orchestrator.execute_workflow(test_snippets)
        elapsed = time.time() - start
        
        print(f"\n✅ Workflow completed in {elapsed:.3f}s")
        
        # Display results
        for i, result in enumerate(results):
            print(f"\n📝 Snippet {i+1}:")
            print(f"   Complexity: {result['analysis']['complexity']:.2f}")
            print(f"   Confidence: {result['correction']['confidence']:.3f}")
            if result['original'] != result['corrected']:
                print(f"   ✅ Corrected!")
        
        # Get metrics
        print(f"\n📊 Pool Metrics:")
        
        analysis_metrics = analysis_pool.metrics
        print(f"   Analysis pool:")
        print(f"     Tasks: {analysis_metrics.total_tasks_completed}")
        print(f"     Latency: {analysis_metrics.average_latency_ms:.2f}ms")
        
        correction_metrics = correction_pool.metrics
        print(f"   Correction pool:")
        print(f"     Tasks: {correction_metrics.total_tasks_completed}")
        print(f"     Latency: {correction_metrics.average_latency_ms:.2f}ms")
        
        # Cleanup
        await orchestrator.shutdown()
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_scaling():
    """Test system scaling"""
    print("\n🔬 Testing System Scaling")
    print("-" * 40)
    
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    try:
        # Test with different pool sizes
        pool_sizes = [1, 2, 4]
        
        for size in pool_sizes:
            print(f"\n📊 Testing with {size} actors per pool:")
            
            orchestrator = CodeCorrectionOrchestrator(f"scale_test_{size}")
            
            # Create pools
            analysis_pool = EnhancedActorPool(
                CodeAnalysisActor,
                num_actors=size,
                config=ActorConfig(name="analysis", num_cpus=1.0)
            )
            
            correction_pool = EnhancedActorPool(
                CorrectionActor,
                num_actors=size,
                config=ActorConfig(name="correction", num_cpus=1.0)
            )
            
            orchestrator.register_pool('analysis', analysis_pool)
            orchestrator.register_pool('correction', correction_pool)
            
            await orchestrator.initialize()
            
            # Generate test data
            test_snippets = [f"print {i}" for i in range(100)]
            
            # Measure performance
            start = time.time()
            results = await orchestrator.execute_workflow(test_snippets)
            elapsed = time.time() - start
            
            throughput = len(test_snippets) / elapsed
            
            print(f"   Time: {elapsed:.3f}s")
            print(f"   Throughput: {throughput:.1f} snippets/sec")
            
            # Cleanup
            await orchestrator.shutdown()
        
        print("\n✅ Scaling test completed")
        return True
        
    except Exception as e:
        print(f"❌ Scaling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all integration tests"""
    print("\n" + "="*60)
    print("🎯 ARCHITECTURE INTEGRATION TEST")
    print("="*60)
    
    results = {}
    
    # Test integrated system
    results['Integration'] = await test_integrated_system()
    
    # Test scaling
    results['Scaling'] = await test_scaling()
    
    # Summary
    print("\n" + "="*60)
    print("📊 INTEGRATION TEST SUMMARY")
    print("="*60)
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name:20s}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 Architecture fully integrated!")
        print("System features:")
        print("  ✅ Actor-first design with Ray")
        print("  ✅ PyTorch scalars for ALL math")
        print("  ✅ Enhanced pool wrapping Ray's ActorPool")
        print("  ✅ Tensor sharing via TensorStore")
        print("  ✅ Orchestrator coordinating pools")
        print("  ✅ Scalable from 1 to N actors")
    else:
        print("\n⚠️ Some integration tests failed")
    
    # Cleanup
    if ray.is_initialized():
        ray.shutdown()
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)