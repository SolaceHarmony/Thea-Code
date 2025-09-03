#!/usr/bin/env python
"""
Thea Code System - Main Entry Point
Production-ready distributed code correction system

Usage:
    python -m thea_code_system.main --directory ./my_project
    python -m thea_code_system.main --file example.js
    python -m thea_code_system.main --config production --workers 16
"""

import argparse
import asyncio
import ray
import sys
import time
from pathlib import Path
from typing import Optional

from .config import get_config, load_config_from_env, ProductionConfig
from .actors.core import OrchestrationActor


async def process_directory(directory: str, config: ProductionConfig) -> None:
    """Process an entire directory"""
    
    print(f"🚀 Processing directory: {directory}")
    print(f"📊 Configuration: {config.max_workers} workers, {config.context_length} context")
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    try:
        # Create orchestrator
        orchestrator = OrchestrationActor.remote(config)
        
        # Process codebase
        start_time = time.time()
        results = await orchestrator.process_codebase.remote(directory)
        
        if results.get('success', False):
            # Generate and display report
            report = await orchestrator.generate_report.remote(results)
            print(report)
            
            processing_time = time.time() - start_time
            print(f"\n⏱️  Total processing time: {processing_time:.2f}s")
            
        else:
            print(f"❌ Processing failed: {results.get('error', 'Unknown error')}")
            sys.exit(1)
            
    finally:
        ray.shutdown()


async def process_file(file_path: str, config: ProductionConfig) -> None:
    """Process a single file"""
    
    print(f"📄 Processing file: {file_path}")
    
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        sys.exit(1)
    
    # For single file, create a temporary directory structure
    temp_dir = Path(file_path).parent
    await process_directory(str(temp_dir), config)


def health_check(config: ProductionConfig) -> None:
    """Run system health check"""
    
    print("🏥 Running system health check...")
    
    # Check Ray
    try:
        ray.init(ignore_reinit_error=True)
        print("✅ Ray initialization: OK")
        ray.shutdown()
    except Exception as e:
        print(f"❌ Ray initialization: FAILED - {e}")
        return
    
    # Check PyTorch
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA: {torch.cuda.device_count()} devices")
        elif torch.backends.mps.is_available():
            print("✅ MPS: Available")
        else:
            print("✅ CPU: Available")
            
    except Exception as e:
        print(f"❌ PyTorch: FAILED - {e}")
        return
    
    # Check model loading
    try:
        from .models.m2bert import M2BertConfig
        config_test = M2BertConfig()
        print("✅ Model configuration: OK")
    except Exception as e:
        print(f"❌ Model configuration: FAILED - {e}")
        return
    
    print("🎉 All health checks passed!")


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description="Thea Code System - Distributed Code Correction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --directory ./my_project
  %(prog)s --file example.js  
  %(prog)s --config production --workers 16
  %(prog)s --health-check
        """
    )
    
    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--directory", "-d",
        help="Directory to process"
    )
    group.add_argument(
        "--file", "-f", 
        help="Single file to process"
    )
    group.add_argument(
        "--health-check",
        action="store_true",
        help="Run system health check"
    )
    
    # Configuration options
    parser.add_argument(
        "--config", "-c",
        choices=["development", "dev", "production", "prod", "edge", "mobile"],
        default="production",
        help="Configuration preset to use"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        help="Number of worker actors (overrides config)"
    )
    parser.add_argument(
        "--context-length",
        type=int,
        help="Context length in tokens (overrides config)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        help="Confidence threshold for fixes (overrides config)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Health check mode
    if args.health_check:
        config = get_config(args.config)
        health_check(config)
        return
    
    # Load configuration
    if any(os_var.startswith("THEA_") for os_var in os.environ):
        config = load_config_from_env()
        print("📋 Configuration loaded from environment variables")
    else:
        config = get_config(args.config)
        print(f"📋 Configuration loaded: {args.config}")
    
    # Apply command line overrides
    if args.workers:
        config.max_workers = args.workers
    if args.context_length:
        config.context_length = args.context_length
    if args.confidence:
        config.confidence_threshold = args.confidence
    if args.verbose:
        config.log_level = "DEBUG"
    
    # Process input
    try:
        if args.directory:
            asyncio.run(process_directory(args.directory, config))
        elif args.file:
            asyncio.run(process_file(args.file, config))
            
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()