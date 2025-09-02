#!/usr/bin/env python
"""
Core Correction Actor
The heart of our distributed system - handles code analysis and fixes

Updated imports and refined architecture
"""

import ray
import torch
import torch.nn as nn
import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# Our internal imports (fixed paths)
from ...models.m2bert.compatibility import M2BertModel, M2BertConfig, load_pretrained_m2bert
from ...utils.patterns.eslint_patterns import CodePattern
from ...config.production import ProductionConfig


@ray.remote(num_gpus=0.25 if torch.cuda.is_available() else 0)
class CodeCorrectionActor:
    """
    Main code correction actor using M2-BERT + patterns
    
    This is our refined architecture:
    - M2-BERT for context understanding (32k tokens)
    - Pattern matching for reliable fixes
    - Async processing throughout
    - Production-ready error handling
    """
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.actor_id = ray.get_runtime_context().get_actor_id()
        
        print(f"🎭 CodeCorrectionActor {self.actor_id} initializing...")
        
        # Device setup with proper optimization
        self.device = self._setup_device()
        
        # Load models and patterns
        self.load_m2bert()
        self.load_patterns()
        
        print(f"✅ Actor {self.actor_id} ready on {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup optimal device configuration"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            # Enable optimizations
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            # CPU optimizations
            torch.set_num_threads(min(8, torch.get_num_threads()))
        
        return device
    
    def load_m2bert(self):
        """Load M2-BERT with all optimizations"""
        
        # Load pretrained M2-BERT (Apache 2.0!)
        self.model, self.config_m2bert = load_pretrained_m2bert(
            self.config.model_path
        )
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Compile for performance
        if self.config.enable_torch_compile and hasattr(torch, 'compile'):
            self.model = torch.compile(
                self.model,
                mode="max-autotune" if self.device.type != "cpu" else "default"
            )
        
        # Add our correction head
        self.correction_head = nn.Sequential(
            nn.Linear(self.config_m2bert.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, len(self.config.supported_rules)),
            nn.Sigmoid()  # For multi-label classification
        ).to(self.device)
        
        # Load tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=True
        )
    
    def load_patterns(self):
        """Load ESLint patterns for reliable fixes"""
        self.patterns = CodePattern.get_all_patterns()
    
    async def analyze_code(self, code: str, file_path: str) -> Dict[str, Any]:
        """
        Analyze code using M2-BERT + patterns
        
        Returns:
            Analysis results with detected issues and confidence scores
        """
        
        start_time = time.perf_counter()
        
        try:
            # Tokenize with full 32k context
            inputs = self.tokenizer(
                code,
                return_tensors="pt",
                max_length=self.config.context_length,
                truncation=True,
                padding=True
            ).to(self.device)
            
            token_count = inputs['attention_mask'].sum().item()
            
            # M2-BERT inference
            with torch.no_grad():
                outputs = self.model(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask']
                )
                
                # Get pooled representation
                hidden_states = outputs['last_hidden_state']
                pooled = hidden_states.mean(dim=1)  # Global average pooling
                
                # Rule prediction
                rule_probs = self.correction_head(pooled).squeeze(0)
            
            inference_time = time.perf_counter() - start_time
            
            # Extract detected rules above threshold
            detected_rules = []
            for i, prob in enumerate(rule_probs):
                if prob > self.config.confidence_threshold:
                    rule_name = self.config.supported_rules[i]
                    detected_rules.append({
                        'rule': rule_name,
                        'confidence': prob.item(),
                        'has_pattern': rule_name in self.patterns
                    })
            
            # Sort by confidence
            detected_rules.sort(key=lambda x: x['confidence'], reverse=True)
            
            return {
                'success': True,
                'file_path': file_path,
                'actor_id': self.actor_id,
                'token_count': token_count,
                'inference_time': inference_time,
                'throughput': token_count / inference_time,
                'detected_rules': detected_rules,
                'model_used': 'M2-BERT-32k',
                'device': str(self.device)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'file_path': file_path,
                'actor_id': self.actor_id
            }
    
    async def fix_code(self, code: str, detected_rules: List[Dict]) -> Dict[str, Any]:
        """
        Apply fixes using pattern matching
        
        ML detects, patterns fix - production reliability!
        """
        
        fixed_code = code
        applied_fixes = []
        
        try:
            for rule_info in detected_rules:
                rule = rule_info['rule']
                confidence = rule_info['confidence']
                
                if rule in self.patterns and rule_info['has_pattern']:
                    pattern = self.patterns[rule]
                    
                    # Apply pattern-based fix
                    original_code = fixed_code
                    fixed_code, fix_count = pattern.apply_fix(fixed_code)
                    
                    if fix_count > 0:
                        applied_fixes.append({
                            'rule': rule,
                            'fixes_applied': fix_count,
                            'confidence': confidence * pattern.reliability,
                            'method': 'pattern'
                        })
            
            return {
                'success': True,
                'original_code': code,
                'fixed_code': fixed_code,
                'applied_fixes': applied_fixes,
                'total_fixes': sum(f['fixes_applied'] for f in applied_fixes),
                'has_changes': fixed_code != code
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'original_code': code
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for the actor"""
        
        try:
            # Quick inference test
            test_code = "if (x = = 5) { return true; }"
            result = await self.analyze_code(test_code, "health_check.js")
            
            return {
                'status': 'healthy' if result['success'] else 'unhealthy',
                'actor_id': self.actor_id,
                'device': str(self.device),
                'model_loaded': hasattr(self, 'model'),
                'patterns_loaded': len(self.patterns),
                'test_inference_time': result.get('inference_time', 0)
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'actor_id': self.actor_id,
                'error': str(e)
            }


# Export for easy importing
__all__ = ['CodeCorrectionActor']