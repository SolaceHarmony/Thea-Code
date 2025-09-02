#!/usr/bin/env python
"""
Core Correction Actor V2
Actor-first architecture with PyTorch scalars for ALL math

Every operation, no matter how trivial, uses PyTorch scalars
"""

import torch
import torch.nn as nn
import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# Use our new wrappers!
from ...wrappers import (
    Actor, 
    async_actor, 
    remote_method,
    scalar, add, sub, mul, div,
    TorchMath, TorchCounter, TorchAccumulator
)
from ...models.m2bert.compatibility import M2BertModel, M2BertConfig, load_pretrained_m2bert
from ...utils.patterns.eslint_patterns import CodePattern
from ...config.production import ProductionConfig


@async_actor(num_gpus=0.25 if torch.cuda.is_available() else 0)
class CodeCorrectionActorV2(Actor):
    """
    Code Correction Actor with actor-first design
    
    ALL math operations use PyTorch scalars:
    - Token counting uses torch tensors
    - Confidence scores are torch tensors
    - Even loop counters are torch tensors
    """
    
    def __init__(self, config: ProductionConfig, name: str = None):
        super().__init__(name)
        self.config = config
        
        print(f"🎭 {self.name} initializing with PyTorch-everywhere philosophy...")
        
        # Use TorchCounter for ALL counting
        self.files_processed = TorchCounter()
        self.errors_found = TorchCounter()
        self.fixes_applied = TorchCounter()
        
        # Use TorchAccumulator for metrics
        self.confidence_accumulator = TorchAccumulator()
        self.inference_time_accumulator = TorchAccumulator()
        
        # Load models and patterns
        self.load_m2bert()
        self.load_patterns()
        
        print(f"✅ {self.name} ready on {self.device}")
    
    def load_m2bert(self) -> None:
        """Load M2-BERT model with 32k context"""
        print(f"📚 Loading M2-BERT-32k model...")
        
        self.model, self.m2bert_config = load_pretrained_m2bert(
            self.config.model_path
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Compile if configured - using torch scalars for config check
        compile_flag = scalar(1 if self.config.enable_torch_compile else 0)
        if TorchMath.gt(compile_flag, scalar(0)):
            self.model = torch.compile(self.model)
            print("⚡ Model compiled with torch.compile()")
    
    def load_patterns(self) -> None:
        """Load ESLint pattern definitions"""
        self.patterns = {}
        
        # Count patterns using torch scalar
        pattern_count = TorchCounter()
        
        for rule_name in self.config.supported_rules:
            pattern = CodePattern.get_pattern(rule_name)
            if pattern:
                self.patterns[rule_name] = pattern
                pattern_count.increment()
        
        # Use torch scalar for comparison
        total_patterns = pattern_count.get()
        print(f"📏 Loaded {total_patterns.item():.0f} ESLint patterns")
    
    @remote_method
    async def analyze_code(self, code: str, file_path: str) -> Dict[str, Any]:
        """
        Analyze code using M2-BERT + patterns
        ALL metrics use PyTorch scalars
        """
        
        # Start timing with torch scalar
        start_time = scalar(time.time())
        
        # Increment file counter
        self.files_processed.increment()
        
        # Tokenize with length as torch scalar
        tokens = self.tokenize(code)
        token_count = scalar(len(tokens))
        
        # Check context length using torch comparison
        max_context = scalar(self.config.context_length)
        if TorchMath.gt(token_count, max_context):
            print(f"⚠️ File exceeds {max_context.item():.0f} tokens")
            token_count = max_context
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(tokens)
            embeddings = outputs.last_hidden_state
        
        # Pattern matching with torch scalar counting
        detected_issues = []
        issue_count = TorchCounter()
        
        for rule_name, pattern in self.patterns.items():
            matches = pattern.find_matches(code)
            
            # Count matches using torch scalar
            match_count = scalar(len(matches))
            if TorchMath.gt(match_count, scalar(0)):
                issue_count.increment(match_count)
                
                for match in matches:
                    # Calculate confidence as torch scalar
                    confidence = self.calculate_confidence(
                        embeddings, match, pattern
                    )
                    self.confidence_accumulator.add(confidence)
                    
                    detected_issues.append({
                        'rule': rule_name,
                        'line': match['line'],
                        'column': match['column'],
                        'confidence': confidence.item(),
                        'fix': pattern.generate_fix(match)
                    })
        
        # Update error counter
        self.errors_found.increment(issue_count.get())
        
        # Calculate inference time using torch scalars
        end_time = scalar(time.time())
        inference_time = sub(end_time, start_time)
        self.inference_time_accumulator.add(inference_time)
        
        return {
            'file_path': file_path,
            'issues': detected_issues,
            'token_count': token_count.item(),
            'inference_time': inference_time.item(),
            'total_issues': issue_count.get().item()
        }
    
    def calculate_confidence(
        self, 
        embeddings: torch.Tensor,
        match: Dict,
        pattern: CodePattern
    ) -> torch.Tensor:
        """
        Calculate confidence score using ONLY torch operations
        Returns torch scalar between 0 and 1
        """
        
        # Base confidence from pattern (as torch scalar)
        base_confidence = scalar(pattern.confidence)
        
        # Context score from embeddings (simplified)
        # Take mean of relevant embeddings
        start_idx = match.get('token_start', 0)
        end_idx = match.get('token_end', start_idx + 1)
        
        if start_idx < embeddings.size(1):
            context_embeddings = embeddings[:, start_idx:end_idx, :]
            
            # Calculate attention-like score using torch ops
            attention_scores = torch.mean(context_embeddings, dim=[1, 2])
            context_score = torch.sigmoid(attention_scores)
        else:
            context_score = scalar(0.5)
        
        # Combine scores using torch operations
        weight_base = scalar(0.7)
        weight_context = scalar(0.3)
        
        final_confidence = add(
            mul(base_confidence, weight_base),
            mul(context_score, weight_context)
        )
        
        # Clamp between 0 and 1
        return TorchMath.clamp(final_confidence, scalar(0), scalar(1))
    
    @remote_method
    async def fix_code(self, code: str, issues: List[Dict]) -> Dict[str, Any]:
        """
        Apply fixes to code based on detected issues
        Uses torch scalars for all counting and metrics
        """
        
        fixed_code = code
        fixes_applied = TorchCounter()
        confidence_threshold = scalar(self.config.confidence_threshold)
        
        # Sort issues by position (reverse order for safe fixing)
        issues_sorted = sorted(issues, key=lambda x: x['line'], reverse=True)
        
        for issue in issues_sorted:
            issue_confidence = scalar(issue['confidence'])
            
            # Only apply high-confidence fixes (torch comparison)
            if TorchMath.ge(issue_confidence, confidence_threshold):
                if issue.get('fix'):
                    fixed_code = self.apply_fix(fixed_code, issue['fix'])
                    fixes_applied.increment()
                    self.fixes_applied.increment()
        
        total_fixes = fixes_applied.get()
        
        return {
            'original_code': code,
            'fixed_code': fixed_code,
            'fixes_applied': total_fixes.item(),
            'confidence_threshold': confidence_threshold.item()
        }
    
    def apply_fix(self, code: str, fix: Dict) -> str:
        """Apply a single fix to code"""
        # This remains string manipulation, but counts use torch
        lines = code.split('\n')
        line_idx = fix['line'] - 1
        
        if 0 <= line_idx < len(lines):
            lines[line_idx] = fix['replacement']
        
        return '\n'.join(lines)
    
    def tokenize(self, code: str) -> torch.Tensor:
        """Tokenize code for M2-BERT"""
        # Simplified tokenization - in production would use proper tokenizer
        tokens = code.split()
        
        # Convert to tensor indices (simplified)
        token_ids = [hash(token) % 30000 for token in tokens]
        
        # Pad/truncate to context length
        max_len = int(self.config.context_length)
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
        else:
            token_ids.extend([0] * (max_len - len(token_ids)))
        
        return torch.tensor(token_ids, device=self.device).unsqueeze(0)
    
    @remote_method
    async def get_metrics(self) -> Dict[str, Any]:
        """
        Get actor metrics - ALL values are torch scalars
        """
        
        # Calculate averages using torch operations
        avg_confidence = self.confidence_accumulator.mean()
        avg_inference_time = self.inference_time_accumulator.mean()
        
        # Calculate fix rate using torch division
        fix_rate = scalar(0)
        if TorchMath.gt(self.errors_found.get(), scalar(0)):
            fix_rate = div(self.fixes_applied.get(), self.errors_found.get())
        
        return {
            'actor': self.name,
            'files_processed': self.files_processed.get().item(),
            'errors_found': self.errors_found.get().item(),
            'fixes_applied': self.fixes_applied.get().item(),
            'avg_confidence': avg_confidence.item() if avg_confidence.numel() > 0 else 0,
            'avg_inference_time': avg_inference_time.item() if avg_inference_time.numel() > 0 else 0,
            'fix_rate': fix_rate.item(),
            'device': str(self.device)
        }
    
    @remote_method
    async def health_check(self) -> Dict[str, Any]:
        """Health check with torch scalar metrics"""
        
        # Calculate memory usage with torch scalars
        if torch.cuda.is_available():
            memory_used = scalar(torch.cuda.memory_allocated() / 1024**3)  # GB
            memory_total = scalar(torch.cuda.memory_reserved() / 1024**3)
        else:
            memory_used = scalar(0)
            memory_total = scalar(0)
        
        # Model size in parameters (torch scalar)
        param_count = scalar(sum(p.numel() for p in self.model.parameters()))
        
        return {
            'actor': self.name,
            'healthy': True,
            'device': str(self.device),
            'model_loaded': self.model is not None,
            'patterns_loaded': len(self.patterns),
            'files_processed': self.files_processed.get().item(),
            'memory_gb': memory_used.item(),
            'model_params_millions': div(param_count, scalar(1e6)).item()
        }
    
    @remote_method  
    async def reset_metrics(self) -> None:
        """Reset all metrics - using torch operations"""
        self.files_processed.reset()
        self.errors_found.reset()
        self.fixes_applied.reset()
        self.confidence_accumulator.reset()
        self.inference_time_accumulator.reset()
        
        print(f"📊 {self.name} metrics reset")