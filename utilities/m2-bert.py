#!/usr/bin/env python3
"""
M2-BERT: Monarch Mixer BERT for TypeScript/ESLint Error Correction
Main entry point for the M2-BERT system

Architecture based on Stanford Hazy Research's Monarch Mixer (M2)
Teacher: Qwen3-Coder:30B
Student: M2-BERT with subquadratic attention
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional
import importlib.util

# Component registry
COMPONENTS = {
    "train-eslint": {
        "module": "m2-bert-eslint-trainer.py",
        "class": "ESLintTrainer",
        "method": "training_loop",
        "description": "Train M2-BERT using ESLint errors with Qwen3 teacher",
        "args": {"max_errors": 100}
    },
    "train-chaos": {
        "module": "m2-bert-chaos-trainer.py",
        "class": "ChaosTrainer",
        "method": "chaos_training_loop",
        "description": "Chaos engineering - corrupt code and learn to repair",
        "args": {"num_sessions": 10, "files_per_session": 5}
    },
    "train-self": {
        "module": "m2-bert-self-supervised.py",
        "class": "SelfSupervisedTrainer",
        "method": "self_supervised_training_loop",
        "description": "Self-supervised learning from linter feedback",
        "args": {}
    },
    "train-continuous": {
        "module": "m2-bert-continuous-learner.py",
        "class": "ContinuousLearner",
        "method": "continuous_learning_loop",
        "description": "Continuous learning with memory replay",
        "args": {"max_iterations": 50}
    },
    "train-semantic": {
        "module": "m2-bert-semantic-chaos-trainer.py",
        "class": "SemanticChaosTrainer",
        "method": "semantic_chaos_loop",
        "description": "Train with semantic validation to prevent degenerate fixes",
        "args": {"num_iterations": 100}
    },
    "generate-diffs": {
        "module": "m2-bert-diff-generator.py",
        "class": "DiffGenerator",
        "method": "training_loop",
        "description": "Generate unified diffs for TypeScript fixes",
        "args": {"max_cycles": 10}
    },
    "validate": {
        "module": "m2-bert-validator.py",
        "class": "SymbioticFixer",
        "method": "run_symbiotic_loop",
        "description": "Validate fixes with linter and Qwen3",
        "args": {}
    },
    "test": {
        "module": "m2-bert-smoke-test.py",
        "class": "SmokeTestRunner",
        "method": "run_all_tests",
        "description": "Run smoke tests to verify system functionality",
        "args": {"quick": False}
    },
    "evaluate": {
        "module": "bert-mad-evaluator.py",
        "class": "MADEvaluator",
        "method": "evaluate_all",
        "description": "Evaluate architecture using MAD synthetic tasks",
        "args": {}
    }
}

def load_module(module_path: str):
    """Dynamically load a Python module"""
    spec = importlib.util.spec_from_file_location("module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def run_component(component_name: str, project_path: str = ".", **kwargs):
    """Run a specific M2-BERT component"""
    if component_name not in COMPONENTS:
        print(f"Unknown component: {component_name}")
        print(f"Available: {', '.join(COMPONENTS.keys())}")
        return False
    
    config = COMPONENTS[component_name]
    print(f"\n{'='*60}")
    print(f"M2-BERT: {config['description']}")
    print(f"{'='*60}\n")
    
    try:
        # Load module
        module = load_module(config['module'])
        
        # Get class and instantiate
        component_class = getattr(module, config['class'])
        
        # Handle different initialization patterns
        if component_name in ['train-eslint', 'train-chaos', 'train-semantic']:
            instance = component_class(project_path)
        elif component_name == 'test':
            instance = component_class(verbose=kwargs.get('verbose', True))
        else:
            instance = component_class()
        
        # Get method and run
        method = getattr(instance, config['method'])
        
        # Merge default args with provided kwargs
        args = {**config['args'], **kwargs}
        
        # Run the component
        result = method(**args)
        
        print(f"\n{'='*60}")
        print("Component execution completed")
        print(f"{'='*60}\n")
        
        return True
        
    except Exception as e:
        print(f"Error running component: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(
        description="M2-BERT: Monarch Mixer BERT for TypeScript/ESLint Error Correction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run smoke tests
  python m2-bert.py test
  
  # Train with ESLint errors
  python m2-bert.py train-eslint --max-errors 50
  
  # Run chaos training
  python m2-bert.py train-chaos --sessions 5
  
  # Generate diffs for current errors
  python m2-bert.py generate-diffs --cycles 3
  
  # List all components
  python m2-bert.py list
        """
    )
    
    parser.add_argument(
        "command",
        nargs="?",
        choices=list(COMPONENTS.keys()) + ["list"],
        help="Component to run"
    )
    
    parser.add_argument(
        "--project",
        default=".",
        help="Project directory to analyze (default: current directory)"
    )
    
    parser.add_argument(
        "--max-errors",
        type=int,
        help="Maximum errors to process (for training commands)"
    )
    
    parser.add_argument(
        "--sessions",
        type=int,
        dest="num_sessions",
        help="Number of training sessions (for chaos training)"
    )
    
    parser.add_argument(
        "--iterations",
        type=int,
        dest="num_iterations",
        help="Number of iterations (for semantic training)"
    )
    
    parser.add_argument(
        "--cycles",
        type=int,
        dest="max_cycles",
        help="Maximum training cycles (for diff generation)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode (for testing)"
    )
    
    args = parser.parse_args()
    
    if not args.command or args.command == "list":
        print("\nM2-BERT Components:")
        print("="*60)
        for name, config in COMPONENTS.items():
            print(f"  {name:18} - {config['description']}")
        print("="*60)
        print("\nUsage: python m2-bert.py <component> [options]")
        print("Example: python m2-bert.py test --quick")
        return
    
    # Build kwargs from args
    kwargs = {}
    if args.max_errors is not None:
        kwargs['max_errors'] = args.max_errors
    if hasattr(args, 'num_sessions') and args.num_sessions is not None:
        kwargs['num_sessions'] = args.num_sessions
    if hasattr(args, 'num_iterations') and args.num_iterations is not None:
        kwargs['num_iterations'] = args.num_iterations
    if hasattr(args, 'max_cycles') and args.max_cycles is not None:
        kwargs['max_cycles'] = args.max_cycles
    if args.verbose:
        kwargs['verbose'] = args.verbose
    if args.quick:
        kwargs['quick'] = args.quick
    
    # Run the component
    success = run_component(args.command, args.project, **kwargs)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()