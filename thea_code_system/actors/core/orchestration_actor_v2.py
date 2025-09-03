#!/usr/bin/env python
"""
Orchestration Actor V2
Actor-first orchestration with PyTorch scalars everywhere

Coordinates worker actors using torch scalars for all operations
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
import time

# Our actor-first wrappers
from ...wrappers import (
    Actor,
    ActorPool,
    async_actor,
    remote_method,
    scalar, add, sub, mul, div,
    TorchMath, TorchCounter, TorchAccumulator
)
from ...config.production import ProductionConfig
from .correction_actor_v2 import CodeCorrectionActorV2


@async_actor(num_cpus=2)
class OrchestrationActorV2(Actor):
    """
    Orchestrator using actor-first design
    
    ALL operations use PyTorch scalars:
    - Worker pool management
    - Load balancing calculations  
    - Performance metrics
    - Even simple counters
    """
    
    def __init__(self, config: ProductionConfig, name: str = None):
        super().__init__(name or "Orchestrator")
        self.config = config
        
        print(f"🎼 {self.name} initializing with {config.max_workers} workers...")
        
        # Create actor pool
        self.worker_pool = ActorPool(
            CodeCorrectionActorV2,
            num_actors=config.max_workers,
            config=config
        )
        
        # Metrics using torch scalars
        self.total_files = TorchCounter()
        self.completed_files = TorchCounter()
        self.failed_files = TorchCounter()
        self.total_issues = TorchCounter()
        self.total_fixes = TorchCounter()
        
        # Timing metrics
        self.processing_times = TorchAccumulator()
        self.queue_times = TorchAccumulator()
        
        # Load balancing with torch
        self.worker_loads = [TorchCounter() for _ in range(config.max_workers)]
        
        print(f"✅ {self.name} ready with {config.max_workers} worker actors")
    
    @remote_method
    async def process_codebase(self, directory: str) -> Dict[str, Any]:
        """
        Process entire codebase using actor pool
        All metrics computed with torch scalars
        """
        
        start_time = scalar(time.time())
        
        # Find all code files
        code_files = self.find_code_files(directory)
        file_count = scalar(len(code_files))
        self.total_files.value = file_count
        
        print(f"📂 Found {file_count.item():.0f} files to process")
        
        # Process files in batches using actor pool
        batch_size = scalar(self.config.batch_size)
        total_batches = TorchMath.ceil(div(file_count, batch_size))
        
        results = []
        current_batch = TorchCounter()
        
        # Process batches
        for i in range(0, len(code_files), int(batch_size.item())):
            current_batch.increment()
            batch_files = code_files[i:i+int(batch_size.item())]
            
            # Calculate progress using torch scalars
            progress = mul(
                div(current_batch.get(), total_batches),
                scalar(100)
            )
            
            print(f"🔄 Processing batch {current_batch.get().item():.0f}/{total_batches.item():.0f} ({progress.item():.1f}%)")
            
            # Process batch using actor pool
            batch_results = await self.process_batch(batch_files)
            results.extend(batch_results)
        
        # Calculate final metrics using torch
        end_time = scalar(time.time())
        total_time = sub(end_time, start_time)
        
        # Aggregate results with torch operations
        success_rate = div(self.completed_files.get(), self.total_files.get())
        avg_issues_per_file = div(self.total_issues.get(), self.completed_files.get())
        fix_rate = div(self.total_fixes.get(), self.total_issues.get())
        
        return {
            'success': True,
            'directory': directory,
            'total_files': self.total_files.get().item(),
            'completed_files': self.completed_files.get().item(),
            'failed_files': self.failed_files.get().item(),
            'total_issues': self.total_issues.get().item(),
            'total_fixes': self.total_fixes.get().item(),
            'success_rate': success_rate.item(),
            'avg_issues_per_file': avg_issues_per_file.item(),
            'fix_rate': fix_rate.item(),
            'total_time': total_time.item(),
            'results': results
        }
    
    async def process_batch(self, files: List[str]) -> List[Dict]:
        """
        Process a batch of files using the actor pool
        Load balancing uses torch scalars
        """
        
        tasks = []
        
        for file_path in files:
            # Read file content
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
            except Exception as e:
                self.failed_files.increment()
                continue
            
            # Get next worker using load balancing
            worker = self.get_balanced_worker()
            
            # Submit task to worker
            task = self.process_file(worker, code, file_path)
            tasks.append(task)
        
        # Wait for all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                self.failed_files.increment()
            else:
                self.completed_files.increment()
                processed_results.append(result)
                
                # Update metrics using torch scalars
                if 'total_issues' in result:
                    self.total_issues.increment(scalar(result['total_issues']))
                if 'fixes_applied' in result:
                    self.total_fixes.increment(scalar(result['fixes_applied']))
        
        return processed_results
    
    async def process_file(self, worker, code: str, file_path: str) -> Dict:
        """
        Process single file with a worker
        Timing uses torch scalars
        """
        
        queue_start = scalar(time.time())
        
        # Analyze code
        analysis = await worker.call('analyze_code', code, file_path)
        
        # Apply fixes if issues found
        if analysis['issues']:
            fix_result = await worker.call('fix_code', code, analysis['issues'])
            analysis['fixes_applied'] = fix_result['fixes_applied']
            analysis['fixed_code'] = fix_result['fixed_code']
            
            # Write fixed code back
            if fix_result['fixes_applied'] > 0:
                await self.write_fixed_code(file_path, fix_result['fixed_code'])
        
        # Calculate timing with torch
        queue_end = scalar(time.time())
        queue_time = sub(queue_end, queue_start)
        self.queue_times.add(queue_time)
        
        return analysis
    
    def get_balanced_worker(self):
        """
        Get worker with lowest load
        Load balancing uses torch scalar comparisons
        """
        
        # Find worker with minimum load
        min_load = self.worker_loads[0].get()
        min_idx = 0
        
        for i in range(1, len(self.worker_loads)):
            load = self.worker_loads[i].get()
            if TorchMath.lt(load, min_load):
                min_load = load
                min_idx = i
        
        # Increment load for selected worker
        self.worker_loads[min_idx].increment()
        
        return self.worker_pool.actors[min_idx]
    
    async def write_fixed_code(self, file_path: str, fixed_code: str) -> None:
        """Write fixed code back to file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_code)
        except Exception as e:
            print(f"❌ Failed to write {file_path}: {e}")
    
    def find_code_files(self, directory: str) -> List[str]:
        """Find all code files in directory"""
        
        path = Path(directory)
        extensions = ['.js', '.jsx', '.ts', '.tsx', '.py']
        
        code_files = []
        for ext in extensions:
            code_files.extend(str(f) for f in path.rglob(f'*{ext}'))
        
        return code_files
    
    @remote_method
    async def get_worker_metrics(self) -> List[Dict]:
        """
        Get metrics from all workers
        Aggregates using torch scalars
        """
        
        metrics = await self.worker_pool.broadcast('get_metrics')
        
        # Aggregate metrics using torch
        total_processed = TorchAccumulator()
        total_errors = TorchAccumulator()
        total_fixes = TorchAccumulator()
        
        for m in metrics:
            total_processed.add(scalar(m['files_processed']))
            total_errors.add(scalar(m['errors_found']))
            total_fixes.add(scalar(m['fixes_applied']))
        
        return {
            'workers': metrics,
            'aggregate': {
                'total_files': total_processed.total().item(),
                'total_errors': total_errors.total().item(),
                'total_fixes': total_fixes.total().item(),
                'avg_files_per_worker': total_processed.mean().item(),
                'avg_errors_per_worker': total_errors.mean().item(),
                'avg_fixes_per_worker': total_fixes.mean().item()
            }
        }
    
    @remote_method
    async def health_check(self) -> Dict[str, Any]:
        """
        Health check for orchestrator and all workers
        All metrics use torch scalars
        """
        
        # Check worker pool health
        pool_health = await self.worker_pool.health_check()
        
        # Calculate orchestrator metrics
        files_per_second = scalar(0)
        if TorchMath.gt(self.processing_times.size(), scalar(0)):
            avg_time = self.processing_times.mean()
            if TorchMath.gt(avg_time, scalar(0)):
                files_per_second = div(scalar(1), avg_time)
        
        # Load balance score (lower is better)
        load_variance = scalar(0)
        if len(self.worker_loads) > 0:
            loads = [l.get() for l in self.worker_loads]
            if len(loads) > 1:
                import torch
                load_tensor = torch.stack(loads)
                load_variance = torch.var(load_tensor)
        
        return {
            'orchestrator': self.name,
            'healthy': True,
            'worker_pool': pool_health,
            'files_processed': self.completed_files.get().item(),
            'files_failed': self.failed_files.get().item(),
            'avg_queue_time': self.queue_times.mean().item() if self.queue_times.size().item() > 0 else 0,
            'files_per_second': files_per_second.item(),
            'load_balance_variance': load_variance.item()
        }
    
    @remote_method
    async def generate_report(self, results: Dict) -> str:
        """
        Generate processing report
        All calculations use torch scalars
        """
        
        # Extract metrics
        total_files = scalar(results['total_files'])
        completed = scalar(results['completed_files'])
        failed = scalar(results['failed_files'])
        issues = scalar(results['total_issues'])
        fixes = scalar(results['total_fixes'])
        total_time = scalar(results['total_time'])
        
        # Calculate rates using torch
        success_rate = mul(div(completed, total_files), scalar(100))
        fix_rate = mul(div(fixes, issues), scalar(100)) if TorchMath.gt(issues, scalar(0)) else scalar(0)
        files_per_minute = mul(div(completed, div(total_time, scalar(60))), scalar(1))
        
        report = f"""
🎯 PROCESSING COMPLETE
{'='*50}

📊 SUMMARY
Files processed: {completed.item():.0f}/{total_files.item():.0f} ({success_rate.item():.1f}% success)
Issues found: {issues.item():.0f}
Fixes applied: {fixes.item():.0f} ({fix_rate.item():.1f}% fix rate)
Failed files: {failed.item():.0f}

⏱️ PERFORMANCE  
Total time: {total_time.item():.2f}s
Processing rate: {files_per_minute.item():.1f} files/minute
Workers used: {self.config.max_workers}

✅ All metrics computed with PyTorch scalars on {str(self.device)}
"""
        
        return report
    
    @remote_method
    async def shutdown(self) -> None:
        """Graceful shutdown of orchestrator and workers"""
        print(f"🛑 Shutting down {self.name}...")
        
        # Shutdown worker pool
        self.worker_pool.shutdown()
        
        # Call parent shutdown
        await super().shutdown()
        
        print(f"✅ {self.name} shutdown complete")