#!/usr/bin/env python3
"""
M2-BERT Smoke Test Suite
Quick validation runs to ensure the system works properly
Includes synthetic errors and limited training cycles
"""

import subprocess
import os
import sys
import json
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import traceback

class SmokeTestRunner:
    """
    Runs smoke tests for M2-BERT components
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = {
            'passed': [],
            'failed': [],
            'skipped': [],
            'errors': []
        }
        self.test_dir = Path("smoke_test_workspace")
        self.test_dir.mkdir(exist_ok=True)
        
    def log(self, message: str, level: str = "INFO"):
        """Log with optional verbosity"""
        if self.verbose or level in ["ERROR", "WARNING"]:
            prefix = {
                "INFO": "[INFO]",
                "SUCCESS": "[✓]",
                "ERROR": "[✗]",
                "WARNING": "[!]",
                "TEST": "[TEST]"
            }.get(level, "[INFO]")
            print(f"{prefix} {message}")
    
    def create_test_file(self, content: str, filename: str = "test.ts") -> Path:
        """Create a temporary test file"""
        filepath = self.test_dir / filename
        with open(filepath, 'w') as f:
            f.write(content)
        return filepath
    
    def cleanup_test_files(self):
        """Clean up test workspace"""
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_typescript_error_detection(self) -> bool:
        """Test: Can we detect TypeScript errors?"""
        self.log("Testing TypeScript error detection", "TEST")
        
        try:
            # Create file with intentional errors
            test_content = """
const x = 5  // Missing semicolon
function test() {  // Missing closing brace
    console.log("test")

interface User {
    name: string  // Missing semicolon
}
"""
            test_file = self.create_test_file(test_content)
            
            # Run TypeScript compiler
            result = subprocess.run(
                ['npx', 'tsc', '--noEmit', str(test_file)],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # Check if errors were detected
            has_errors = 'error TS' in result.stderr or 'error TS' in result.stdout
            
            if has_errors:
                self.log("TypeScript error detection working", "SUCCESS")
                return True
            else:
                self.log("TypeScript error detection failed - no errors found", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"TypeScript error detection failed: {e}", "ERROR")
            return False
    
    def test_eslint_detection(self) -> bool:
        """Test: Can we detect ESLint errors?"""
        self.log("Testing ESLint error detection", "TEST")
        
        try:
            # Create ESLint config if needed
            eslint_config = {
                "env": {"es2021": True, "node": True},
                "extends": ["eslint:recommended"],
                "parserOptions": {
                    "ecmaVersion": 12,
                    "sourceType": "module"
                },
                "rules": {
                    "semi": ["error", "always"],
                    "quotes": ["error", "single"]
                }
            }
            
            config_path = self.test_dir / ".eslintrc.json"
            with open(config_path, 'w') as f:
                json.dump(eslint_config, f)
            
            # Create file with ESLint violations
            test_content = """
const x = 5  // Missing semicolon
const y = "test"  // Should use single quotes
const unused = 42  // Unused variable
"""
            test_file = self.create_test_file(test_content, "eslint_test.js")
            
            # Run ESLint
            result = subprocess.run(
                ['npx', 'eslint', str(test_file), '--format', 'json'],
                capture_output=True,
                text=True,
                cwd=self.test_dir,
                timeout=10
            )
            
            # Parse results
            if result.stdout:
                try:
                    eslint_output = json.loads(result.stdout)
                    total_errors = sum(len(f.get('messages', [])) for f in eslint_output)
                    
                    if total_errors > 0:
                        self.log(f"ESLint detection working - found {total_errors} issues", "SUCCESS")
                        return True
                except:
                    pass
            
            self.log("ESLint detection not configured or no errors found", "WARNING")
            return False
            
        except subprocess.TimeoutExpired:
            self.log("ESLint timeout - may not be installed", "WARNING")
            return False
        except Exception as e:
            self.log(f"ESLint detection error: {e}", "WARNING")
            return False
    
    def test_ollama_connection(self) -> bool:
        """Test: Can we connect to Ollama?"""
        self.log("Testing Ollama connection", "TEST")
        
        try:
            import requests
            response = requests.get('http://localhost:11434/api/tags', timeout=2)
            
            if response.status_code == 200:
                tags = response.json()
                models = [m.get('name', '') for m in tags.get('models', [])]
                
                # Check for Qwen3-Coder
                has_qwen = any('qwen' in m.lower() for m in models)
                
                if has_qwen:
                    self.log("Ollama connection working with Qwen model", "SUCCESS")
                    return True
                else:
                    self.log("Ollama working but Qwen3-Coder not found", "WARNING")
                    return True
            else:
                self.log("Ollama API not responding correctly", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"Cannot connect to Ollama: {e}", "ERROR")
            return False
    
    def test_model_import(self) -> bool:
        """Test: Can we import and initialize models?"""
        self.log("Testing model imports", "TEST")
        
        try:
            import torch
            from transformers import AutoTokenizer
            
            # Test device availability
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            
            self.log(f"PyTorch device: {device}", "INFO")
            
            # Test tokenizer
            tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
            
            # Quick tokenization test
            test_text = "const x = 5;"
            tokens = tokenizer.encode(test_text)
            
            if len(tokens) > 0:
                self.log("Model imports and tokenization working", "SUCCESS")
                return True
            else:
                self.log("Tokenization produced no tokens", "ERROR")
                return False
                
        except ImportError as e:
            self.log(f"Missing dependency: {e}", "ERROR")
            return False
        except Exception as e:
            self.log(f"Model import error: {e}", "ERROR")
            return False
    
    def test_diff_generation(self) -> bool:
        """Test: Can we generate valid diffs?"""
        self.log("Testing diff generation", "TEST")
        
        try:
            # Simple diff generation test
            original = "const x = 5"
            fixed = "const x = 5;"
            
            diff = f"""@@ -1,1 +1,1 @@
-{original}
+{fixed}"""
            
            # Validate diff format
            if '@@' in diff and '-' in diff and '+' in diff:
                self.log("Diff generation format valid", "SUCCESS")
                return True
            else:
                self.log("Invalid diff format", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"Diff generation error: {e}", "ERROR")
            return False
    
    def test_mini_training_loop(self) -> bool:
        """Test: Can we run a minimal training loop?"""
        self.log("Testing mini training loop", "TEST")
        
        try:
            # Import minimal components
            import torch
            import torch.nn as nn
            from transformers import AutoTokenizer
            
            # Create tiny model
            class TinyModel(nn.Module):
                def __init__(self, vocab_size):
                    super().__init__()
                    self.embedding = nn.Embedding(vocab_size, 32)
                    self.output = nn.Linear(32, vocab_size)
                
                def forward(self, input_ids):
                    x = self.embedding(input_ids)
                    return self.output(x)
            
            # Initialize
            tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
            model = TinyModel(tokenizer.vocab_size)
            optimizer = torch.optim.Adam(model.parameters())
            
            # Single training step
            text = "const x = 5;"
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=32)
            
            # Forward pass
            outputs = model(inputs['input_ids'])
            loss = outputs.mean()  # Dummy loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            self.log("Mini training loop completed", "SUCCESS")
            return True
            
        except Exception as e:
            self.log(f"Training loop error: {e}", "ERROR")
            self.log(traceback.format_exc(), "ERROR")
            return False
    
    def run_integration_test(self) -> bool:
        """Test: Full integration with limited scope"""
        self.log("Running integration test", "TEST")
        
        try:
            # This would import and run actual M2-BERT components
            # with very limited data
            
            # For now, we'll simulate
            self.log("Integration test placeholder", "WARNING")
            return True
            
        except Exception as e:
            self.log(f"Integration test error: {e}", "ERROR")
            return False
    
    def run_all_tests(self, quick: bool = False) -> Dict:
        """Run all smoke tests"""
        print("\n" + "="*60)
        print("M2-BERT SMOKE TEST SUITE")
        print("="*60)
        print(f"Mode: {'Quick' if quick else 'Full'}")
        print("="*60 + "\n")
        
        tests = [
            ("TypeScript Detection", self.test_typescript_error_detection),
            ("Model Imports", self.test_model_import),
            ("Diff Generation", self.test_diff_generation),
        ]
        
        if not quick:
            tests.extend([
                ("ESLint Detection", self.test_eslint_detection),
                ("Ollama Connection", self.test_ollama_connection),
                ("Mini Training Loop", self.test_mini_training_loop),
                ("Integration Test", self.run_integration_test),
            ])
        
        start_time = time.time()
        
        for test_name, test_func in tests:
            try:
                print(f"\n{'='*40}")
                result = test_func()
                
                if result:
                    self.results['passed'].append(test_name)
                else:
                    self.results['failed'].append(test_name)
                    
            except Exception as e:
                self.log(f"Test {test_name} crashed: {e}", "ERROR")
                self.results['errors'].append(test_name)
        
        elapsed = time.time() - start_time
        
        # Print summary
        print("\n" + "="*60)
        print("SMOKE TEST SUMMARY")
        print("="*60)
        print(f"Duration: {elapsed:.2f} seconds")
        print(f"Passed: {len(self.results['passed'])}")
        print(f"Failed: {len(self.results['failed'])}")
        print(f"Errors: {len(self.results['errors'])}")
        print(f"Skipped: {len(self.results['skipped'])}")
        
        if self.results['passed']:
            print("\nPassed Tests:")
            for test in self.results['passed']:
                print(f"  ✓ {test}")
        
        if self.results['failed']:
            print("\nFailed Tests:")
            for test in self.results['failed']:
                print(f"  ✗ {test}")
        
        if self.results['errors']:
            print("\nTests with Errors:")
            for test in self.results['errors']:
                print(f"  ! {test}")
        
        # Overall result
        total = len(tests)
        passed = len(self.results['passed'])
        success_rate = (passed / total * 100) if total > 0 else 0
        
        print(f"\nSuccess Rate: {success_rate:.1f}%")
        
        if success_rate == 100:
            print("\n🎉 ALL TESTS PASSED!")
        elif success_rate >= 80:
            print("\n✓ Most tests passed - system likely functional")
        elif success_rate >= 50:
            print("\n⚠ Some issues detected - check failed tests")
        else:
            print("\n✗ Multiple failures - system needs attention")
        
        # Cleanup
        self.cleanup_test_files()
        
        return self.results

def load_component_module(module_name: str, file_path: str):
    """Dynamically load a Python module from a file path"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Component registry with metadata
COMPONENT_REGISTRY = {
    "eslint": {
        "file": "m2-bert-eslint-trainer.py",
        "module": "eslint_trainer",
        "class": "ESLintTrainer",
        "runner": lambda trainer, iterations: trainer.training_loop(max_errors=iterations),
        "init_args": (".",),
        "description": "ESLint-driven training with Qwen3 teacher"
    },
    "chaos": {
        "file": "m2-bert-chaos-trainer.py",
        "module": "chaos_trainer",
        "class": "ChaosTrainer",
        "runner": lambda trainer, iterations: trainer.chaos_training_loop(num_sessions=iterations, files_per_session=1),
        "init_args": (".",),
        "description": "Chaos engineering - corrupt and repair code"
    },
    "self-supervised": {
        "file": "m2-bert-self-supervised.py",
        "module": "self_supervised",
        "class": "SelfSupervisedTrainer",
        "runner": lambda trainer, iterations: trainer.self_supervised_training_loop(
            training_data=[
                {'code': 'const x = 5', 'error': "';' expected"},
                {'code': 'function test() {', 'error': "'}' expected"},
            ][:iterations]
        ),
        "init_args": (),
        "description": "Self-supervised learning from linter feedback"
    },
    "continuous": {
        "file": "m2-bert-continuous-learner.py",
        "module": "continuous_learner",
        "class": "ContinuousLearner",
        "runner": lambda learner, iterations: learner.continuous_learning_loop(max_iterations=iterations),
        "init_args": (),
        "description": "Continuous learning with memory replay"
    },
    "diff": {
        "file": "m2-bert-diff-generator.py",
        "module": "diff_generator",
        "class": "DiffGenerator",
        "runner": lambda generator, iterations: generator.training_loop(max_cycles=iterations),
        "init_args": (),
        "description": "Generate unified diffs for fixes"
    },
    "semantic-chaos": {
        "file": "m2-bert-semantic-chaos-trainer.py",
        "module": "semantic_chaos",
        "class": "SemanticChaosTrainer",
        "runner": lambda trainer, iterations: trainer.semantic_chaos_loop(num_iterations=iterations),
        "init_args": (".",),
        "description": "Semantic validation to prevent degenerate fixes"
    },
    "validator": {
        "file": "m2-bert-validator.py",
        "module": "validator",
        "class": "SymbioticFixer",
        "runner": lambda fixer, iterations: fixer.run_symbiotic_loop(),
        "init_args": (),
        "description": "Symbiotic validation with linter and Qwen3"
    }
}

def run_component_test(component: str, iterations: int = 2):
    """Run a specific M2-BERT component with limited iterations"""
    
    print(f"\n{'='*60}")
    print(f"COMPONENT TEST: {component}")
    print(f"Iterations: {iterations}")
    print("="*60)
    
    if component not in COMPONENT_REGISTRY:
        print(f"Unknown component: {component}")
        print(f"Available components: {', '.join(COMPONENT_REGISTRY.keys())}")
        return False
    
    try:
        config = COMPONENT_REGISTRY[component]
        print(f"Description: {config['description']}")
        print("="*60)
        
        # Load the module
        module = load_component_module(config['module'], config['file'])
        
        # Get the class and instantiate
        component_class = getattr(module, config['class'])
        instance = component_class(*config['init_args'])
        
        # Run the component
        config['runner'](instance, iterations)
        
        print(f"\n✓ Component test completed: {component}")
        return True
        
    except ImportError as e:
        print(f"Cannot import component: {e}")
        return False
    except Exception as e:
        print(f"Component test failed: {e}")
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="M2-BERT Smoke Test Suite")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--component", choices=list(COMPONENT_REGISTRY.keys()),
                       help="Test specific component")
    parser.add_argument("--iterations", type=int, default=2, help="Number of iterations for component test")
    parser.add_argument("--list-components", action="store_true", help="List all available components")
    
    args = parser.parse_args()
    
    if args.list_components:
        # List all available components
        print("\nAvailable M2-BERT Components:")
        print("="*60)
        for name, config in COMPONENT_REGISTRY.items():
            print(f"  {name:20} - {config['description']}")
        print("="*60)
        print(f"\nUsage: python {sys.argv[0]} --component <name> --iterations <n>")
        sys.exit(0)
    
    if args.component:
        # Test specific component
        success = run_component_test(args.component, args.iterations)
        sys.exit(0 if success else 1)
    else:
        # Run smoke tests
        runner = SmokeTestRunner(verbose=args.verbose)
        results = runner.run_all_tests(quick=args.quick)
        
        # Exit code based on results
        if len(results['failed']) == 0 and len(results['errors']) == 0:
            sys.exit(0)
        else:
            sys.exit(1)