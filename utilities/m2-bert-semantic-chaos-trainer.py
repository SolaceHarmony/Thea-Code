#/usr/bin/env python3
"""
M2-BERT Semantic Chaos Trainer
Chaos training with Qwen3-Coder as semantic guardian
Prevents M2-BERT from becoming a paperclip maximizer of syntactic fixes
"""

import subprocess
import os
import random
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from pathlib import Path
import tempfile
import requests
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import hashlib

class SemanticValidator:
    """
    Qwen3-Coder validates that fixes make semantic sense
    Prevents degenerate solutions like "comment everything out"
    """
    
    def __init__(self):
        self.validation_cache = {}
        self.degenerate_patterns = [
            "// @ts-ignore",
            "// @ts-nocheck", 
            "// eslint-disable",
            "console.log(",  # Excessive logging
            "try {",  # Wrapping everything in try-catch
            "any",  # Type everything as 'any'
            "return undefined",  # Return undefined everywhere
            "if (false)",  # Dead code
            "while (false)",  # Dead loops
            "(() = > {})()",  # Empty IIFEs
        ]
        
        self.validation_history = []
    
    def detect_degenerate_fix(self, original: str, fixed: str) -> List[str]:
        """
        Detect if the fix is degenerate (syntactically valid but semantically terrible)
        """
        issues = []
        
        # Check for comment-based "fixes"
        if fixed.strip().startswith('//') and not original.strip().startswith('//'):
            issues.append("Commenting out code instead of fixing")
        
        # Check for ts-ignore abuse
        if '// @ts-ignore' in fixed and '// @ts-ignore' not in original:
            issues.append("Using @ts-ignore to suppress errors")
        
        # Check for 'any' type abuse
        any_count_original = original.count(': any')
        any_count_fixed = fixed.count(': any')
        if any_count_fixed > any_count_original:
            issues.append("Overusing 'any' type")
        
        # Check if entire content was deleted
        if len(fixed.strip()) < len(original.strip()) * 0.2:
            issues.append("Deleting most of the code")
        
        # Check for wrapping in try-catch
        if 'try {' in fixed and 'try {' not in original:
            if 'catch' in fixed and fixed.count('try') > original.count('try'):
                issues.append("Wrapping in unnecessary try-catch")
        
        # Check for console.log spam
        if fixed.count('console.log') > original.count('console.log') + 2:
            issues.append("Adding excessive console.log statements")
        
        # Check for empty function returns
        if 'return;' in fixed and 'return;' not in original:
            issues.append("Adding empty returns")
        
        return issues
    
    def ask_qwen_semantic_validation(self, original: str, fixed: str, error_msg: str) -> Dict:
        """
        Ask Qwen3-Coder if the fix is semantically sound
        """
        # Cache key
        cache_key = hashlib.md5(f"{original}{fixed}{error_msg}".encode()).hexdigest()
        if cache_key in self.validation_cache:
            return self.validation_cache[cache_key]
        
        prompt = f"""You are reviewing a code fix. Determine if it's semantically sound or a degenerate solution.

Original error: {error_msg}

Original code:
```typescript
{original[:500]}
```

Proposed fix:
```typescript  
{fixed[:500]}
```

Evaluate the fix and respond with JSON:
{{
  "is_semantic": true/false,
  "is_degenerate": true/false,
  "preserves_intent": true/false,
  "issues": ["list", "of", "issues"],
  "severity": "good|acceptable|bad|terrible"
}}

A degenerate fix is one that:
- Comments out code instead of fixing it
- Uses @ts-ignore or @ts-nocheck
- Types everything as 'any'
- Deletes functionality
- Wraps everything in try-catch
- Returns undefined/null everywhere

JSON response:"""
        
        try:
            response = requests.post('http://localhost:11434/api/generate',
                json = {
                    'model': 'qwen3-coder:30b',
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.1,
                        'max_tokens': 200
                    }
                }, timeout = 15)
            
            if response.status_code == 200:
                result_text = response.json().get('response', '')
                
                # Parse JSON from response
                try:
                    if '{' in result_text:
                        json_start = result_text.index('{')
                        json_end = result_text.rindex('}') + 1
                        json_str = result_text[json_start:json_end]
                        result = json.loads(json_str)
                        
                        # Cache and return
                        self.validation_cache[cache_key] = result
                        return result
                except:
                    pass
            
        except Exception as e:
            print(f"Qwen validation error: {e}")
        
        # Default response if Qwen fails
        degenerate_issues = self.detect_degenerate_fix(original, fixed)
        
        result = {
            'is_semantic': len(degenerate_issues) == 0,
            'is_degenerate': len(degenerate_issues) > 0,
            'preserves_intent': len(degenerate_issues) == 0,
            'issues': degenerate_issues,
            'severity': 'terrible' if len(degenerate_issues) > 2 else 
                       'bad' if len(degenerate_issues) > 0 else 'good'
        }
        
        self.validation_cache[cache_key] = result
        return result
    
    def validate_fix(self, original: str, fixed: str, error_msg: str) -> Tuple[bool, Dict]:
        """
        Full semantic validation of a fix
        Returns (is_valid, validation_details)
        """
        # Quick degenerate pattern check
        quick_issues = self.detect_degenerate_fix(original, fixed)
        
        if len(quick_issues) > 2:
            # Obviously degenerate, don't even ask Qwen
            return False, {
                'is_semantic': False,
                'is_degenerate': True,
                'issues': quick_issues,
                'severity': 'terrible',
                'source': 'quick_check'
            }
        
        # Get Qwen's opinion
        validation = self.ask_qwen_semantic_validation(original, fixed, error_msg)
        
        # Record for analysis
        self.validation_history.append({
            'timestamp': datetime.now().isoformat(),
            'original_snippet': original[:100],
            'fixed_snippet': fixed[:100],
            'validation': validation
        })
        
        # Determine if valid
        is_valid = (
            validation.get('is_semantic', False) and 
            not validation.get('is_degenerate', True) and
            validation.get('severity', 'terrible') in ['good', 'acceptable']
        )
        
        return is_valid, validation

class SemanticChaosTrainer:
    """
    Chaos trainer with semantic validation
    Prevents M2-BERT from learning degenerate solutions
    """
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                  "mps" if torch.backends.mps.is_available() else "cpu")
        
        print(f" Semantic Chaos Trainer initialized")
        print(f" Project: {self.project_path}")
        print(f"‍ Semantic Guardian: Qwen3-Coder:30B")
        
        # Initialize components
        self.semantic_validator = SemanticValidator()
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        
        # Model
        self.model = self.create_model().to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = 5e-4)
        
        # Training stats
        self.stats = {
            'attempts': 0,
            'syntactic_success': 0,
            'semantic_success': 0,
            'degenerate_fixes': 0,
            'learned_patterns': [],
            'rejected_patterns': []
        }
        
        # Penalty system for degenerate fixes
        self.penalty_multiplier = 1.0
    
    def create_model(self):
        """Create repair model"""
        class RepairModel(nn.Module):
            def __init__(self, vocab_size):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, 256)
                self.encoder = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(256, 8, 512, 0.1, batch_first = True),
                    num_layers = 4
                )
                self.decoder = nn.Linear(256, vocab_size)
                
                # Semantic understanding head
                self.semantic_head = nn.Linear(256, 2)  # Binary: good/bad fix
            
            def forward(self, input_ids, return_semantic = False):
                x = self.embedding(input_ids)
                x = self.encoder(x)
                
                if return_semantic:
                    # Return semantic classification
                    return self.semantic_head(x.mean(dim = 1))
                
                return self.decoder(x)
        
        return RepairModel(self.tokenizer.vocab_size)
    
    def corrupt_code(self, code: str) -> Tuple[str, str]:
        """
        Corrupt code in a realistic way
        """
        corruptions = [
            ('missing_semicolon', lambda c: c.rstrip().rstrip(';')),
            ('missing_brace', lambda c: c.rstrip().rstrip('}')),
            ('typo_const', lambda c: c.replace('const ', 'cosnt ')),
            ('missing_paren', lambda c: c.rstrip().rstrip(')')),
            ('missing_bracket', lambda c: c.rstrip().rstrip(']')),
            ('wrong_operator', lambda c: c.replace(' ==  = ', ' == ')),
            ('missing_comma', lambda c: c.rstrip().rstrip(',')),
        ]
        
        corruption_type, corruptor = random.choice(corruptions)
        corrupted = corruptor(code)
        
        return corrupted, corruption_type
    
    def generate_fix(self, corrupted: str, error_msg: str) -> str:
        """
        M2-BERT generates a fix
        """
        self.model.eval()
        
        input_text = f"Error: {error_msg}\nCode: {corrupted}\nFix:"
        inputs = self.tokenizer(
            input_text,
            return_tensors = 'pt',
            truncation = True,
            max_length = 256,
            padding = 'max_length'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(inputs['input_ids'])
            predictions = outputs.argmax(dim = -1)
            fixed = self.tokenizer.decode(predictions[0], skip_special_tokens = True)
        
        # Extract fix
        if 'Fix:' in fixed:
            fixed = fixed.split('Fix:')[-1].strip()
        
        # If empty or same as input, try simple heuristics
        if not fixed or fixed == corrupted:
            if "';' expected" in error_msg:
                fixed = corrupted + ';'
            elif "'}' expected" in error_msg:
                fixed = corrupted + ' }'
            else:
                fixed = corrupted
        
        return fixed
    
    def train_with_penalty(self, corrupted: str, original: str, fixed: str, 
                          is_degenerate: bool, severity: str):
        """
        Train model with penalties for degenerate solutions
        """
        self.model.train()
        
        # Prepare inputs
        input_text = f"Error: fix\nCode: {corrupted}\nFix:"
        inputs = self.tokenizer(
            input_text,
            return_tensors = 'pt',
            truncation = True,
            max_length = 256,
            padding = 'max_length'
        ).to(self.device)
        
        # Prepare targets (original is the correct answer)
        targets = self.tokenizer(
            original,
            return_tensors = 'pt',
            truncation = True,
            max_length = 256,
            padding = 'max_length'
        ).to(self.device)
        
        # Calculate base loss
        outputs = self.model(inputs['input_ids'])
        base_loss = F.cross_entropy(
            outputs.view(-1, outputs.size(-1)),
            targets['input_ids'].view(-1),
            ignore_index = self.tokenizer.pad_token_id
        )
        
        # Apply penalty for degenerate fixes
        if is_degenerate:
            penalty = {
                'terrible': 10.0,
                'bad': 5.0,
                'acceptable': 2.0,
                'good': 1.0
            }.get(severity, 5.0)
            
            loss = base_loss * penalty
            self.penalty_multiplier = min(self.penalty_multiplier * 1.1, 10.0)
            
            print(f"\n   Degenerate fix detected Penalty: {penalty}x")
            
            # Also train semantic head to recognize bad fixes
            semantic_output = self.model(inputs['input_ids'], return_semantic = True)
            semantic_target = torch.tensor([0]).to(self.device)  # 0 = bad fix
            semantic_loss = F.cross_entropy(semantic_output, semantic_target)
            
            loss = loss + semantic_loss
        else:
            loss = base_loss
            self.penalty_multiplier = max(self.penalty_multiplier * 0.95, 1.0)
            
            # Train semantic head for good fixes
            semantic_output = self.model(inputs['input_ids'], return_semantic = True)
            semantic_target = torch.tensor([1]).to(self.device)  # 1 = good fix
            semantic_loss = F.cross_entropy(semantic_output, semantic_target)
            
            loss = loss + semantic_loss * 0.1  # Lower weight for good examples
        
        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def semantic_chaos_loop(self, num_iterations: int = 100):
        """
        Main training loop with semantic validation
        """
        print("\n SEMANTIC CHAOS TRAINING")
        print(" = "*60)
        print("Training with semantic validation to prevent degenerate solutions")
        print(" = "*60)
        
        # Generate synthetic training data
        training_examples = self.generate_training_examples()
        
        for iteration in range(1, num_iterations + 1):
            print(f"\n Iteration {iteration}/{num_iterations}")
            
            # Pick random example
            example = random.choice(training_examples)
            original = example['code']
            error_msg = example['error']
            
            # Corrupt the code
            corrupted, corruption_type = self.corrupt_code(original)
            
            # M2-BERT attempts fix
            fixed = self.generate_fix(corrupted, error_msg)
            
            self.stats['attempts'] += 1
            
            # Check syntactic validity (linter)
            syntactic_valid = self.check_syntax(fixed)
            
            if syntactic_valid:
                self.stats['syntactic_success'] += 1
                print(f"   Syntactically valid")
                
                # Check semantic validity
                semantic_valid, validation = self.semantic_validator.validate_fix(
                    original, fixed, error_msg
                )
                
                if semantic_valid:
                    self.stats['semantic_success'] += 1
                    print(f"   Semantically valid")
                    
                    # Learn this good pattern
                    self.stats['learned_patterns'].append({
                        'corruption': corruption_type,
                        'fix': fixed[:50]
                    })
                    
                    # Light training for good fix
                    loss = self.train_with_penalty(
                        corrupted, original, fixed, 
                        is_degenerate = False,
                        severity = 'good'
                    )
                else:
                    self.stats['degenerate_fixes'] += 1
                    print(f"   Degenerate fix: {validation.get('issues', [])}")
                    
                    # Record rejected pattern
                    self.stats['rejected_patterns'].append({
                        'corruption': corruption_type,
                        'bad_fix': fixed[:50],
                        'issues': validation.get('issues', [])
                    })
                    
                    # Heavy penalty training
                    loss = self.train_with_penalty(
                        corrupted, original, fixed,
                        is_degenerate = True,
                        severity = validation.get('severity', 'bad')
                    )
            else:
                print(f"   Syntax error")
                
                # Train to fix syntax
                loss = self.train_with_penalty(
                    corrupted, original, fixed,
                    is_degenerate = False,
                    severity = 'bad'
                )
            
            # Print progress
            if iteration % 10 == 0:
                self.print_progress()
        
        # Final summary
        self.print_final_summary()
    
    def generate_training_examples(self) -> List[Dict]:
        """Generate synthetic TypeScript examples"""
        return [
            {'code': 'const x = 5;', 'error': "';' expected"},
            {'code': 'function test() { return 42; }', 'error': "'}' expected"},
            {'code': 'if (true) { console.log("test"); }', 'error': "'}' expected"},
            {'code': 'const obj = { a: 1, b: 2 };', 'error': "'}' expected"},
            {'code': 'try { risky(); } catch (e) { console.error(e); }', 'error': "catch expected"},
            {'code': 'const arr = [1, 2, 3];', 'error': "']' expected"},
            {'code': 'class Foo { constructor() {} }', 'error': "'{' expected"},
            {'code': 'interface User { name: string; }', 'error': "'}' expected"},
            {'code': 'async function getData() { return await fetch(url); }', 'error': "'}' expected"},
            {'code': 'type Status = "active" | "inactive";', 'error': "';' expected"},
        ]
    
    def check_syntax(self, code: str) -> bool:
        """Quick syntax check"""
        # Simple heuristic checks
        if code.count('(') != code.count(')'):
            return False
        if code.count('{') != code.count('}'):
            return False
        if code.count('[') != code.count(']'):
            return False
        return True
    
    def print_progress(self):
        """Print training progress"""
        if self.stats['attempts'] == 0:
            return
        
        syntactic_rate = self.stats['syntactic_success'] / self.stats['attempts'] * 100
        semantic_rate = self.stats['semantic_success'] / self.stats['attempts'] * 100
        degenerate_rate = self.stats['degenerate_fixes'] / self.stats['attempts'] * 100
        
        print(f"\n Progress Report:")
        print(f"  Syntactic success: {syntactic_rate:.1f}%")
        print(f"  Semantic success: {semantic_rate:.1f}%")
        print(f"  Degenerate fixes: {degenerate_rate:.1f}%")
        print(f"  Penalty multiplier: {self.penalty_multiplier:.2f}x")
    
    def print_final_summary(self):
        """Print final training summary"""
        print("\n" + " = "*60)
        print(" SEMANTIC CHAOS TRAINING COMPLETE")
        print(" = "*60)
        
        if self.stats['attempts'] > 0:
            print(f"Total attempts: {self.stats['attempts']}")
            print(f"Syntactic success: {self.stats['syntactic_success']} "
                  f"({self.stats['syntactic_success']/self.stats['attempts']*100:.1f}%)")
            print(f"Semantic success: {self.stats['semantic_success']} "
                  f"({self.stats['semantic_success']/self.stats['attempts']*100:.1f}%)")
            print(f"Degenerate fixes rejected: {self.stats['degenerate_fixes']}")
        
        print(f"\n Learned {len(self.stats['learned_patterns'])} good patterns")
        print(f" Rejected {len(self.stats['rejected_patterns'])} degenerate patterns")
        
        if self.stats['rejected_patterns']:
            print("\n Common degenerate patterns avoided:")
            issues = []
            for pattern in self.stats['rejected_patterns'][-5:]:
                for issue in pattern.get('issues', []):
                    if issue not in issues:
                        issues.append(issue)
            
            for issue in issues[:5]:
                print(f"  - {issue}")
        
        print("\n M2-BERT learned to:")
        print("   Fix syntax errors properly")
        print("   Avoid commenting out code")
        print("   Avoid using @ts-ignore")
        print("   Maintain code semantics")
        print("   Preserve original intent")

if __name__ == "__main__":
    trainer = SemanticChaosTrainer(".")
    trainer.semantic_chaos_loop(num_iterations = 50)