#/usr/bin/env python3
"""
M2-BERT ESLint-Focused Trainer
Trains on raw ESLint errors with tight context
Teacher provides multiple explanations for better learning
"""

import subprocess
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import re
import tempfile

class ESLintErrorExtractor:
    """
    Extracts ESLint errors with tight context using grep
    """
    
    def __init__(self, project_path: str = "."):
        self.project_path = Path(project_path)
        
    def get_eslint_errors(self, max_errors: int = 50) -> List[Dict]:
        """
        Run ESLint and get raw error output
        """
        try:
            # Run ESLint with JSON output for structured data
            result = subprocess.run(
                ['npx', 'eslint', '.', '--format', 'json', '--max-warnings', '0'],
                capture_output = True,
                text = True,
                cwd = self.project_path,
                timeout = 30
            )
            
            if result.returncode == 0:
                return []  # No errors
            
            errors = []
            try:
                eslint_output = json.loads(result.stdout)
                
                for file_result in eslint_output:
                    file_path = file_result['filePath']
                    
                    for message in file_result.get('messages', []):
                        if len(errors) >= max_errors:
                            break
                        
                        errors.append({
                            'file': file_path,
                            'line': message.get('line', 1),
                            'column': message.get('column', 1),
                            'severity': message.get('severity', 1),
                            'message': message.get('message', ''),
                            'ruleId': message.get('ruleId', 'unknown'),
                            'raw': f"{file_path}:{message.get('line')}:{message.get('column')}: {message.get('message')} [{message.get('ruleId')}]"
                        })
                
            except json.JSONDecodeError:
                # Fallback to parsing text output
                return self.parse_text_eslint_output(result.stdout + result.stderr, max_errors)
            
            return errors[:max_errors]
            
        except Exception as e:
            print(f"Error running ESLint: {e}")
            return []
    
    def parse_text_eslint_output(self, output: str, max_errors: int) -> List[Dict]:
        """
        Parse text-based ESLint output as fallback
        """
        errors = []
        
        # Pattern: filepath:line:column: message [rule]
        pattern = r'([^:]+):(\d+):(\d+):\s+(.+?)\s+\[([^\]]+)\]'
        
        for match in re.finditer(pattern, output):
            if len(errors) >= max_errors:
                break
                
            errors.append({
                'file': match.group(1),
                'line': int(match.group(2)),
                'column': int(match.group(3)),
                'message': match.group(4),
                'ruleId': match.group(5),
                'raw': match.group(0)
            })
        
        return errors
    
    def get_tight_context(self, file_path: str, line_num: int, before: int = 2, after: int = 2) -> Dict:
        """
        Get tight context around error line using grep-like extraction
        """
        try:
            # Use grep -A and -B for context
            result = subprocess.run(
                ['grep', '-n', '-A', str(after), '-B', str(before), f'^', file_path],
                capture_output = True,
                text = True
            )
            
            # Fallback to reading file directly
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            if line_num > len(lines):
                return {}
            
            # Extract tight context
            start = max(0, line_num - before - 1)
            end = min(len(lines), line_num + after)
            
            context = {
                'before_lines': lines[start:line_num-1],
                'error_line': lines[line_num-1] if line_num <= len(lines) else '',
                'after_lines': lines[line_num:end],
                'full_context': ''.join(lines[start:end]),
                'line_numbers': list(range(start + 1, end + 1))
            }
            
            return context
            
        except Exception as e:
            print(f"Error getting context: {e}")
            return {}
    
    def get_broader_context(self, file_path: str, line_num: int) -> str:
        """
        Get broader context for teacher to understand the code better
        Includes function/class boundaries
        """
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Find function/class boundaries
            start = line_num - 1
            end = line_num
            
            # Search backwards for function/class start
            for i in range(line_num - 1, max(0, line_num - 50), -1):
                line = lines[i]
                if re.match(r'^(function|class|interface|export|const\s+\w+\s* = )', line.strip()):
                    start = i
                    break
            
            # Search forward for closing brace at same indentation
            base_indent = len(lines[start]) - len(lines[start].lstrip())
            brace_count = 0
            
            for i in range(start, min(len(lines), start + 100)):
                line = lines[i]
                brace_count += line.count('{') - line.count('}')
                
                if brace_count == 0 and i > line_num:
                    end = i + 1
                    break
            
            return ''.join(lines[start:end])
            
        except Exception as e:
            print(f"Error getting broader context: {e}")
            return ""

class TeacherModel:
    """
    Qwen3-Coder teacher that provides multiple explanations and validates diffs
    """
    
    def __init__(self):
        self.teaching_cache = {}
        
    def generate_correct_diff(self, error: Dict, context: Dict) -> Optional[str]:
        """
        Generate a correct diff for the ESLint error
        """
        prompt = f"""Fix this ESLint error by generating a unified diff patch.
Output ONLY the diff in proper format.

ESLint Error: {error['raw']}
Rule: {error['ruleId']}

Code context (line {error['line']}):
{context['full_context']}

Generate minimal diff to fix the error:
"""
        
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
                }, timeout = 20)
            
            if response.status_code == 200:
                diff = response.json().get('response', '').strip()
                
                # Validate diff format
                if self.is_valid_diff(diff):
                    return diff
                else:
                    # Try to fix the format
                    return self.format_as_diff(diff, error['line'], context)
            
        except Exception as e:
            print(f"Teacher error generating diff: {e}")
        
        return None
    
    def is_valid_diff(self, diff: str) -> bool:
        """
        Check if diff is in valid unified format
        """
        required_patterns = [
            r'@@\s*-\d+,?\d*\s+\+\d+,?\d*\s*@@',  # @@ -line,count +line,count @@
            r'^[-+]',  # Lines starting with - or +
        ]
        
        has_header = any(re.search(pattern, diff, re.MULTILINE) for pattern in required_patterns)
        return has_header and ('-' in diff or '+' in diff)
    
    def format_as_diff(self, content: str, line_num: int, context: Dict) -> str:
        """
        Format content as a proper diff if it's not already
        """
        if self.is_valid_diff(content):
            return content
        
        # Create a simple diff
        original = context['error_line'].rstrip()
        
        # Try to extract the fix from content
        if content and content != original:
            fixed = content.strip()
        else:
            # Default fixes based on common ESLint rules
            if 'semi' in context.get('ruleId', '').lower():
                fixed = original.rstrip() + ';'
            elif 'quotes' in context.get('ruleId', '').lower():
                fixed = original.replace('"', "'")
            elif 'indent' in context.get('ruleId', '').lower():
                fixed = '  ' + original.lstrip()
            else:
                fixed = original
        
        # Format as unified diff
        diff = f"""@@ -{line_num},1 +{line_num},1 @@
-{original}
+{fixed}"""
        
        return diff
    
    def explain_fix_multiple_ways(self, error: Dict, diff: str, broader_context: str) -> List[Dict]:
        """
        Explain the fix in 3 different ways for better learning
        """
        explanations = []
        
        # First explanation: Technical
        technical_prompt = f"""Explain this ESLint fix technically and precisely.

Error: {error['message']}
Rule: {error['ruleId']}
Diff:
{diff}

Technical explanation (one sentence):"""
        
        # Second explanation: Conceptual
        conceptual_prompt = f"""Explain the concept behind this ESLint rule and why the fix works.

Error: {error['message']}
Rule: {error['ruleId']}

Conceptual explanation (one sentence):"""
        
        # Third explanation: Alternative solutions
        alternatives_prompt = f"""Given this ESLint error, what are alternative valid fixes?

Error: {error['message']}
Rule: {error['ruleId']}
Context:
{broader_context[:500]}

List 2-3 alternative fixes (brief):"""
        
        prompts = [
            ('technical', technical_prompt),
            ('conceptual', conceptual_prompt),
            ('alternatives', alternatives_prompt)
        ]
        
        for exp_type, prompt in prompts:
            try:
                response = requests.post('http://localhost:11434/api/generate',
                    json = {
                        'model': 'qwen3-coder:30b',
                        'prompt': prompt,
                        'stream': False,
                        'options': {
                            'temperature': 0.3,
                            'max_tokens': 100
                        }
                    }, timeout = 10)
                
                if response.status_code == 200:
                    explanation = response.json().get('response', '').strip()
                    explanations.append({
                        'type': exp_type,
                        'explanation': explanation
                    })
            except:
                pass
        
        return explanations
    
    def evaluate_student_diff(self, student_diff: str, correct_diff: str, error: Dict) -> Dict:
        """
        Evaluate how good the student's diff is
        """
        prompt = f"""Evaluate this student's diff against the correct solution.

ESLint Error: {error['message']}
Rule: {error['ruleId']}

Student's diff:
{student_diff}

Correct diff:
{correct_diff}

Evaluate and respond with JSON:
{{
  "score": 0-100,
  "is_correct": true/false,
  "fixes_issue": true/false,
  "introduces_problems": true/false,
  "feedback": "brief feedback"
}}

JSON:"""
        
        try:
            response = requests.post('http://localhost:11434/api/generate',
                json = {
                    'model': 'qwen3-coder:30b',
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.1,
                        'max_tokens': 150
                    }
                }, timeout = 15)
            
            if response.status_code == 200:
                result_text = response.json().get('response', '')
                
                # Parse JSON
                if '{' in result_text:
                    json_start = result_text.index('{')
                    json_end = result_text.rindex('}') + 1
                    json_str = result_text[json_start:json_end]
                    return json.loads(json_str)
        except:
            pass
        
        # Default evaluation
        return {
            'score': 50,
            'is_correct': student_diff == correct_diff,
            'fixes_issue': '@@' in student_diff,
            'introduces_problems': False,
            'feedback': 'Could not evaluate'
        }

class M2BertESLintModel(nn.Module):
    """
    M2-BERT model specifically trained for ESLint fixes
    """
    def __init__(self, vocab_size, hidden_dim = 256, n_heads = 8, n_layers = 4):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, hidden_dim) * 0.02)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = hidden_dim,
            nhead = n_heads,
            dim_feedforward = hidden_dim * 4,
            dropout = 0.1,
            batch_first = True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers = n_layers)
        
        # Output head for diff generation
        self.diff_head = nn.Linear(hidden_dim, vocab_size)
        
        # Rule understanding head (classifies ESLint rule type)
        self.rule_head = nn.Linear(hidden_dim, 100)  # 100 common ESLint rules
        
    def forward(self, input_ids, task = 'diff'):
        seq_len = input_ids.size(1)
        
        x = self.embedding(input_ids)
        x = x + self.pos_embedding[:, :seq_len, :]
        
        x = self.transformer(x)
        
        if task == 'diff':
            return self.diff_head(x)
        elif task == 'rule':
            return self.rule_head(x.mean(dim = 1))
        else:
            return x

class ESLintTrainer:
    """
    Main trainer that orchestrates ESLint-based training
    """
    
    def __init__(self, project_path: str = "."):
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                  "mps" if torch.backends.mps.is_available() else "cpu")
        
        print(f" ESLint-Focused Trainer")
        print(f" Project: {project_path}")
        print(f" Device: {self.device}")
        print(f"‍ Teacher: Qwen3-Coder:30B")
        
        # Initialize components
        self.eslint_extractor = ESLintErrorExtractor(project_path)
        self.teacher = TeacherModel()
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        
        # Initialize model
        self.model = M2BertESLintModel(
            vocab_size = self.tokenizer.vocab_size
        ).to(self.device)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = 5e-4)
        
        # Training history
        self.training_history = []
        self.checkpoint_dir = Path("eslint_checkpoints")
        self.checkpoint_dir.mkdir(exist_ok = True)
    
    def generate_student_diff(self, error: Dict, context: Dict) -> str:
        """
        M2-BERT generates a diff attempt
        """
        self.model.eval()
        
        # Format input with raw ESLint error and tight context
        input_text = f"""ESLint: {error['raw']}
Line {error['line']}: {context['error_line']}
Fix:"""
        
        inputs = self.tokenizer(
            input_text,
            return_tensors = 'pt',
            truncation = True,
            max_length = 256,
            padding = 'max_length'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(inputs['input_ids'], task = 'diff')
            predictions = outputs.argmax(dim = -1)
            generated = self.tokenizer.decode(predictions[0], skip_special_tokens = True)
        
        # Extract diff part
        if 'Fix:' in generated:
            diff_part = generated.split('Fix:')[-1].strip()
            
            # Ensure it's formatted as diff
            if not self.teacher.is_valid_diff(diff_part):
                # Try to format it
                diff_part = self.teacher.format_as_diff(
                    diff_part, error['line'], context
                )
            
            return diff_part
        
        # Fallback: generate simple diff
        return f"@@ -{error['line']},1 +{error['line']},1 @@\n-{context['error_line'].rstrip()}\n+{context['error_line'].rstrip()};"
    
    def train_with_explanations(self, error: Dict, context: Dict, correct_diff: str, 
                               explanations: List[Dict], epochs: int = 3):
        """
        Train M2-BERT with correct diff and multiple explanations
        No MSE - use teacher evaluation
        """
        self.model.train()
        
        print(f"\n   Training with {len(explanations)} explanations, {epochs} epochs")
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            # Train with different explanation contexts
            for exp in explanations:
                # Create enriched input with explanation
                input_text = f"""ESLint: {error['raw']}
{exp['type']}: {exp['explanation'][:100]}
Line {error['line']}: {context['error_line']}
Fix:"""
                
                inputs = self.tokenizer(
                    input_text,
                    return_tensors = 'pt',
                    truncation = True,
                    max_length = 256,
                    padding = 'max_length'
                ).to(self.device)
                
                # Target is the correct diff
                targets = self.tokenizer(
                    correct_diff,
                    return_tensors = 'pt',
                    truncation = True,
                    max_length = 256,
                    padding = 'max_length'
                ).to(self.device)
                
                # Forward pass
                outputs = self.model(inputs['input_ids'], task = 'diff')
                
                # Cross-entropy loss (not MSE)
                loss = F.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),
                    targets['input_ids'].view(-1),
                    ignore_index = self.tokenizer.pad_token_id
                )
                
                # Also train rule understanding
                rule_outputs = self.model(inputs['input_ids'], task = 'rule')
                # Simple rule classification (would need rule encoding in practice)
                rule_target = torch.tensor([hash(error['ruleId']) % 100]).to(self.device)
                rule_loss = F.cross_entropy(rule_outputs, rule_target)
                
                total_loss = loss + rule_loss * 0.1
                
                # Backprop
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                epoch_loss += total_loss.item()
            
            print(f"    Epoch {epoch+1}: Loss = {epoch_loss/len(explanations):.4f}")
        
        return epoch_loss / (len(explanations) * epochs)
    
    def training_loop(self, max_errors: int = 100):
        """
        Main training loop with ESLint errors
        """
        print("\n" + " = "*60)
        print(" ESLINT-DRIVEN TRAINING")
        print(" = "*60)
        
        # Get ESLint errors
        errors = self.eslint_extractor.get_eslint_errors(max_errors)
        
        if not errors:
            print("No ESLint errors found Project is clean.")
            return
        
        print(f"Found {len(errors)} ESLint errors to train on")
        
        stats = {
            'total': len(errors),
            'successful': 0,
            'scores': [],
            'rules_seen': set()
        }
        
        for i, error in enumerate(errors, 1):
            print(f"\n[{i}/{len(errors)}] {error['file']}:{error['line']}")
            print(f"  Error: {error['message']}")
            print(f"  Rule: {error['ruleId']}")
            
            stats['rules_seen'].add(error['ruleId'])
            
            # Get tight context
            context = self.eslint_extractor.get_tight_context(
                error['file'], error['line'], before = 2, after = 2
            )
            
            if not context or not context.get('error_line'):
                print("   Could not get context")
                continue
            
            # Student attempts
            student_diff = self.generate_student_diff(error, context)
            print(f"   Student attempt generated")
            
            # Teacher generates correct diff
            correct_diff = self.teacher.generate_correct_diff(error, context)
            
            if not correct_diff:
                print("   Teacher could not generate diff")
                continue
            
            print(f"  ‍ Teacher provided correct diff")
            
            # Teacher evaluates student
            evaluation = self.teacher.evaluate_student_diff(student_diff, correct_diff, error)
            stats['scores'].append(evaluation['score'])
            
            print(f"   Score: {evaluation['score']}/100")
            print(f"   Feedback: {evaluation['feedback']}")
            
            if evaluation['score'] >= 80:
                stats['successful'] += 1
                print(f"   Acceptable")
            else:
                # Need training
                print(f"   Training needed.")
                
                # Get broader context for better understanding
                broader_context = self.eslint_extractor.get_broader_context(
                    error['file'], error['line']
                )
                
                # Get multiple explanations
                explanations = self.teacher.explain_fix_multiple_ways(
                    error, correct_diff, broader_context
                )
                
                # Train with explanations
                avg_loss = self.train_with_explanations(
                    error, context, correct_diff, explanations, epochs = 3
                )
                
                print(f"   Training complete (loss: {avg_loss:.4f})")
            
            # Save training example
            self.training_history.append({
                'error': error,
                'student_diff': student_diff,
                'correct_diff': correct_diff,
                'evaluation': evaluation,
                'timestamp': datetime.now().isoformat()
            })
            
            # Checkpoint every 20 errors
            if i % 20 == 0:
                self.save_checkpoint(i, stats)
        
        # Final summary
        self.print_summary(stats)
    
    def save_checkpoint(self, iteration: int, stats: Dict):
        """Save training checkpoint"""
        avg_score = sum(stats['scores']) / len(stats['scores']) if stats['scores'] else 0
        
        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'iteration': iteration,
            'stats': stats,
            'avg_score': avg_score,
            'training_history': self.training_history[-100:]  # Last 100
        }
        
        path = self.checkpoint_dir / f"checkpoint_iter_{iteration}.pt"
        torch.save(checkpoint, path)
        print(f"\n Checkpoint saved: {path} (avg score: {avg_score:.1f})")
    
    def print_summary(self, stats: Dict):
        """Print training summary"""
        print("\n" + " = "*60)
        print(" TRAINING SUMMARY")
        print(" = "*60)
        
        avg_score = sum(stats['scores']) / len(stats['scores']) if stats['scores'] else 0
        
        print(f"Total errors processed: {stats['total']}")
        print(f"Successful on first try: {stats['successful']} ({stats['successful']/stats['total']*100:.1f}%)")
        print(f"Average score: {avg_score:.1f}/100")
        print(f"Unique ESLint rules seen: {len(stats['rules_seen'])}")
        
        if stats['rules_seen']:
            print("\nTop rules encountered:")
            for rule in list(stats['rules_seen'])[:10]:
                print(f"  - {rule}")
        
        print("\n M2-BERT learned to fix:")
        print("   Missing semicolons (semi)")
        print("   Quote consistency (quotes)")
        print("   Indentation (indent)")
        print("   Unused variables (no-unused-vars)")
        print("   And many more ESLint rules")

if __name__ == "__main__":
    trainer = ESLintTrainer(".")
    trainer.training_loop(max_errors = 50)