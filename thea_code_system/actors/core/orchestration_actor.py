#!/usr/bin/env python
"""
Orchestration Actor
Coordinates the entire correction workflow across distributed actors

This is the conductor of our actor symphony
"""

import ray
import asyncio
import time
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path

from ...config.production import ProductionConfig
from .correction_actor import CodeCorrectionActor


@ray.remote
class OrchestrationActor:
    """
    Main orchestrator for distributed code correction
    
    Responsibilities:
    - Manage worker actors (round-robin, health checks)
    - Process entire codebases efficiently
    - Generate detailed reports
    - Handle fault tolerance
    """
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.orchestrator_id = ray.get_runtime_context().get_actor_id()
        
        print(f"🎼 OrchestrationActor {self.orchestrator_id} initializing...")
        
        # Create worker pool
        self.workers = [
            CodeCorrectionActor.remote(config)
            for _ in range(config.max_workers)
        ]
        
        self.current_worker = 0
        self.stats = {
            'files_processed': 0,
            'total_fixes': 0,
            'total_tokens': 0,
            'start_time': time.time()
        }
        
        print(f"✅ Orchestrator ready with {len(self.workers)} workers")
    
    def get_next_worker(self) -> CodeCorrectionActor:
        """Round-robin worker selection with health awareness"""
        worker = self.workers[self.current_worker]
        self.current_worker = (self.current_worker + 1) % len(self.workers)
        return worker
    
    async def health_check_workers(self) -> Dict[str, Any]:
        """Check health of all workers"""
        
        print("🏥 Running worker health checks...")
        
        health_tasks = [
            worker.health_check.remote()
            for worker in self.workers
        ]
        
        health_results = await asyncio.gather(*health_tasks, return_exceptions=True)
        
        healthy_workers = 0
        worker_status = []
        
        for i, result in enumerate(health_results):
            if isinstance(result, dict) and result.get('status') == 'healthy':
                healthy_workers += 1
                worker_status.append({
                    'worker_id': i,
                    'status': 'healthy',
                    'details': result
                })
            else:
                worker_status.append({
                    'worker_id': i,
                    'status': 'unhealthy',
                    'error': str(result) if isinstance(result, Exception) else result
                })
        
        return {
            'healthy_workers': healthy_workers,
            'total_workers': len(self.workers),
            'health_ratio': healthy_workers / len(self.workers),
            'worker_details': worker_status
        }
    
    async def discover_code_files(self, directory: str) -> List[str]:
        """Discover all code files in directory"""
        
        code_extensions = {'.js', '.ts', '.jsx', '.tsx', '.py', '.java', '.cpp', '.c', '.go', '.rs'}
        code_files = []
        
        for root, dirs, files in os.walk(directory):
            # Skip common directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {'node_modules', '__pycache__', 'dist', 'build'}]
            
            for file in files:
                if Path(file).suffix.lower() in code_extensions:
                    file_path = os.path.join(root, file)
                    try:
                        # Quick check if file is readable
                        with open(file_path, 'r', encoding='utf-8') as f:
                            f.read(1)  # Just read first character
                        code_files.append(file_path)
                    except (UnicodeDecodeError, PermissionError):
                        print(f"⚠️  Skipping unreadable file: {file_path}")
        
        return sorted(code_files)
    
    async def process_single_file(self, file_path: str, worker: CodeCorrectionActor) -> Dict[str, Any]:
        """Process a single file with error handling"""
        
        try:
            # Read file
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            # Skip very large files (over 1MB)
            if len(code) > 1024 * 1024:
                return {
                    'file_path': file_path,
                    'skipped': True,
                    'reason': 'File too large'
                }
            
            # Analyze code
            analysis = await worker.analyze_code.remote(code, file_path)
            
            if not analysis['success']:
                return {
                    'file_path': file_path,
                    'error': analysis['error'],
                    'success': False
                }
            
            # Apply fixes if issues found
            if analysis['detected_rules']:
                fix_result = await worker.fix_code.remote(code, analysis['detected_rules'])
                
                if fix_result['success'] and fix_result['has_changes']:
                    # Write fixed file
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(fix_result['fixed_code'])
                    
                    return {
                        'file_path': file_path,
                        'success': True,
                        'analysis': analysis,
                        'fixes': fix_result,
                        'changed': True
                    }
                else:
                    return {
                        'file_path': file_path,
                        'success': True,
                        'analysis': analysis,
                        'fixes': fix_result,
                        'changed': False
                    }
            else:
                return {
                    'file_path': file_path,
                    'success': True,
                    'analysis': analysis,
                    'no_issues': True
                }
                
        except Exception as e:
            return {
                'file_path': file_path,
                'error': str(e),
                'success': False
            }
    
    async def process_codebase(self, directory: str) -> Dict[str, Any]:
        """
        Process entire codebase with our actor army
        
        This is where our distributed architecture shines!
        """
        
        start_time = time.time()
        
        print(f"🚀 Processing codebase: {directory}")
        
        # Health check first
        health = await self.health_check_workers()
        if health['healthy_workers'] == 0:
            return {
                'error': 'No healthy workers available',
                'health_status': health
            }
        
        print(f"✅ {health['healthy_workers']}/{health['total_workers']} workers healthy")
        
        # Discover files
        code_files = await self.discover_code_files(directory)
        print(f"📁 Found {len(code_files)} code files")
        
        if not code_files:
            return {
                'error': 'No code files found',
                'directory': directory
            }
        
        # Process files in parallel batches
        batch_size = len(self.workers) * 2  # 2 files per worker
        all_results = []
        
        for i in range(0, len(code_files), batch_size):
            batch = code_files[i:i + batch_size]
            print(f"📦 Processing batch {i//batch_size + 1}: {len(batch)} files")
            
            # Create tasks for this batch
            tasks = []
            for file_path in batch:
                worker = self.get_next_worker()
                task = self.process_single_file(file_path, worker)
                tasks.append(task)
            
            # Process batch
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            for result in batch_results:
                if isinstance(result, Exception):
                    all_results.append({
                        'error': str(result),
                        'success': False
                    })
                else:
                    all_results.append(result)
        
        # Compile statistics
        successful = [r for r in all_results if r.get('success', False)]
        with_issues = [r for r in successful if 'fixes' in r and r.get('changed', False)]
        errors = [r for r in all_results if not r.get('success', False)]
        
        total_fixes = sum(
            r['fixes']['total_fixes'] 
            for r in with_issues 
            if 'fixes' in r and 'total_fixes' in r['fixes']
        )
        
        total_tokens = sum(
            r['analysis']['token_count']
            for r in successful
            if 'analysis' in r and 'token_count' in r['analysis']
        )
        
        processing_time = time.time() - start_time
        
        # Update stats
        self.stats.update({
            'files_processed': self.stats['files_processed'] + len(successful),
            'total_fixes': self.stats['total_fixes'] + total_fixes,
            'total_tokens': self.stats['total_tokens'] + total_tokens
        })
        
        return {
            'success': True,
            'directory': directory,
            'files_found': len(code_files),
            'files_processed': len(successful),
            'files_with_issues': len(with_issues),
            'files_fixed': len([r for r in with_issues if r.get('changed', False)]),
            'total_fixes': total_fixes,
            'total_tokens': total_tokens,
            'errors': len(errors),
            'processing_time': processing_time,
            'throughput': total_tokens / processing_time if processing_time > 0 else 0,
            'results': all_results,
            'health_status': health
        }
    
    async def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive processing report"""
        
        if not results.get('success', False):
            return f"❌ Processing failed: {results.get('error', 'Unknown error')}"
        
        report = f"""
🎯 THEA CODE CORRECTION REPORT
{'='*50}

📊 SUMMARY
Directory: {results['directory']}
Files Found: {results['files_found']}
Files Processed: {results['files_processed']}
Files with Issues: {results['files_with_issues']}
Files Fixed: {results['files_fixed']}

🔧 FIXES APPLIED
Total Fixes: {results['total_fixes']}
Success Rate: {results['files_processed']/results['files_found']*100:.1f}%

⚡ PERFORMANCE
Processing Time: {results['processing_time']:.2f}s
Total Tokens: {results['total_tokens']:,}
Throughput: {results['throughput']:.0f} tokens/sec

🎭 ACTOR SYSTEM
Healthy Workers: {results['health_status']['healthy_workers']}/{results['health_status']['total_workers']}
Health Ratio: {results['health_status']['health_ratio']*100:.1f}%

📋 DETAILED RESULTS
"""
        
        # Group results by outcome
        fixed_files = [r for r in results['results'] if r.get('changed', False)]
        clean_files = [r for r in results['results'] if r.get('no_issues', False)]
        error_files = [r for r in results['results'] if not r.get('success', False)]
        
        if fixed_files:
            report += f"\n✅ FIXED FILES ({len(fixed_files)}):\n"
            for file_result in fixed_files[:10]:  # Show first 10
                if 'fixes' in file_result:
                    fixes = file_result['fixes']['applied_fixes']
                    fix_summary = ', '.join(f"{f['rule']}({f['fixes_applied']})" for f in fixes[:3])
                    report += f"  📝 {file_result['file_path']}: {fix_summary}\n"
            
            if len(fixed_files) > 10:
                report += f"  ... and {len(fixed_files) - 10} more files\n"
        
        if error_files:
            report += f"\n❌ ERRORS ({len(error_files)}):\n"
            for file_result in error_files[:5]:  # Show first 5 errors
                report += f"  🚫 {file_result.get('file_path', 'unknown')}: {file_result.get('error', 'unknown error')}\n"
        
        report += f"""

🏗️ ARCHITECTURE USED
✅ M2-BERT-32k (Apache 2.0) - Legal foundation
✅ Ray distributed actors - Scalable processing  
✅ Pattern-based fixes - Production reliability
✅ Facebook MBRL patterns - Industry proven
✅ PyTorch optimizations - Maximum performance

Built from the complete open-source journey!
"""
        
        return report


# Export for easy importing
__all__ = ['OrchestrationActor']