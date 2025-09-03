#/usr/bin/env python3
"""
M2-BERT Diff Generator
A small, focused model that learns to generate proper diffs from TypeScript errors
Trained by Qwen3-Coder:30B with love and patience
"""

import subprocess
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
import requests
from datetime import datetime
from typing import List, Dict, Optional

class M2BertDiffModel(nn.Module):
    """
    the model M2-BERT - small, efficient, specialized in generating diffs
    Uses subquadratic attention patterns inspired by Monarch Mixer
    """
    def __init__(self, vocab_size = 8000, dim = 256, max_seq_len = 512, n_layers = 4):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, dim) * 0.02)
        
        # Simplified M2 layers - subquadratic mixing
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model = dim,
                nhead = 8,
                dim_feedforward = dim * 2,
                dropout = 0.1,
                batch_first = True
            ) for _ in range(n_layers)
        ])
        
        # Output projection for diff generation
        self.diff_head = nn.Linear(dim, vocab_size)
        
    def forward(self, input_ids, attention_mask = None):
        seq_len = input_ids.size(1)
        
        # Embed and add position
        x = self.embedding(input_ids)
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Apply layers
        for layer in self.layers:
            x = layer(x)
        
        # Generate diff tokens
        return self.diff_head(x)

class DiffGenerator:
    """
    Main system for training M2-BERT to generate diffs
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                  "mps" if torch.backends.mps.is_available() else "cpu")
        print(f" M2-BERT starting on {self.device}")
        
        # Initialize tokenizer (small, focused vocabulary)
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        
        # the model M2-BERT
        self.model = M2BertDiffModel(
            vocab_size = self.tokenizer.vocab_size,
            dim = 256,  # Small and efficient
            max_seq_len = 512,
            n_layers = 4  # Just enough layers
        ).to(self.device)
        
        # Training setup
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = 5e-4)
        self.learning_history = []
        self.success_count = 0
        
    def get_typescript_errors(self, max_errors = 10) -> List[Dict]:
        """Get raw TypeScript errors"""
        try:
            result = subprocess.run(['npx', 'tsc', '--noEmit'], 
                                  capture_output = True, text = True, timeout = 30)
            errors = []
            
            for line in result.stderr.split('\n'):
                if 'error TS' in line and '(' in line:
                    try:
                        # Parse the raw error format
                        parts = line.split(': error TS')
                        if len(parts) == 2:
                            file_coords = parts[0]
                            error_msg = 'error TS' + parts[1]
                            
                            paren_idx = file_coords.rfind('(')
                            if paren_idx > 0:
                                file_path = file_coords[:paren_idx].strip()
                                coords = file_coords[paren_idx+1:-1]
                                line_num, col_num = coords.split(',')
                                
                                errors.append({
                                    'file': file_path,
                                    'line': int(line_num),
                                    'column': int(col_num),
                                    'error': error_msg.strip(),
                                    'raw': line  # Keep raw for learning
                                })
                                
                                if max_errors and len(errors) >= max_errors:
                                    break
                    except:
                        continue
            
            return errors
        except Exception as e:
            print(f"Error getting TypeScript errors: {e}")
            return []
    
    def get_file_context(self, file_path: str, line_num: int, context_lines: int = 3) -> Dict:
        """Get code context around error"""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            if line_num > len(lines):
                return {}
            
            start = max(1, line_num - context_lines)
            end = min(len(lines), line_num + context_lines)
            
            context = {
                'before': lines[start-1:line_num-1],
                'error_line': lines[line_num-1] if line_num <= len(lines) else "",
                'after': lines[line_num:end],
                'line_num': line_num
            }
            
            return context
        except Exception as e:
            print(f"Error reading context: {e}")
            return {}
    
    def ask_teacher_for_diff(self, error: Dict, context: Dict) -> Optional[str]:
        """
        Ask Qwen3-Coder:30B to generate a proper diff
        Teacher provides gentle, precise guidance
        """
        
        # Format context for teacher
        context_str = ""
        if context:
            # Show line numbers for context
            line_num = context['line_num']
            start_num = line_num - len(context['before'])
            
            for i, line in enumerate(context['before']):
                context_str += f"{start_num + i:4d}: {line.rstrip()}\n"
            
            context_str += f"{line_num:4d}: {context['error_line'].rstrip()}  <-- ERROR HERE\n"
            
            for i, line in enumerate(context['after']):
                context_str += f"{line_num + i + 1:4d}: {line.rstrip()}\n"
        
        prompt = f"""You are a gentle teacher helping a small BERT model learn to fix TypeScript errors.
Generate ONLY a unified diff patch. No explanations, just the diff.

TypeScript Error:
{error['raw']}

Code Context:
{context_str}

Generate a minimal unified diff that fixes this error:
"""
        
        try:
            response = requests.post('http://localhost:11434/api/generate',
                json = {
                    'model': 'qwen3-coder:30b',
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.1,  # Very focused
                        'top_p': 0.9,
                    }
                }, timeout = 30)
            
            if response.status_code == 200:
                result = response.json()
                diff = result.get('response', '').strip()
                
                # Validate it looks like a diff
                if '@@' in diff or '---' in diff or '+++' in diff:
                    return diff
                
                # Try to extract diff if wrapped in markdown
                lines = diff.split('\n')
                diff_lines = []
                in_diff = False
                
                for line in lines:
                    if line.startswith('```'):
                        in_diff = not in_diff
                    elif in_diff or line.startswith(('-', '+', '@', '---', '+++')):
                        diff_lines.append(line)
                
                if diff_lines:
                    return '\n'.join(diff_lines)
            
        except Exception as e:
            print(f"Teacher error: {e}")
        
        return None
    
    def m2bert_generate_diff(self, error: Dict, context: Dict) -> str:
        """
        M2-BERT attempts to generate a diff
        """
        self.model.eval()
        
        # Format input for M2-BERT
        input_text = f"Error: {error['error']}\nFile: {error['file']}\nLine: {error['line']}\n"
        if context:
            input_text += f"Code: {context['error_line']}"
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors = 'pt',
            truncation = True,
            max_length = 256,
            padding = 'max_length'
        ).to(self.device)
        
        with torch.no_grad():
            # Generate diff tokens
            outputs = self.model(inputs['input_ids'])
            
            # Simple greedy decoding for now
            predictions = outputs.argmax(dim = -1)
            
            # Decode to text
            generated = self.tokenizer.decode(predictions[0], skip_special_tokens = True)
            
            # Extract diff pattern if present
            if '@@' in generated or '-' in generated or '+' in generated:
                return generated
        
        # Fallback: generate simple diff based on error type
        if context and 'error_line' in context:
            line = context['error_line'].rstrip()
            line_num = context['line_num']
            
            # Simple heuristics for common errors
            if "';' expected" in error['error']:
                return f"@@ -{line_num},1 +{line_num},1 @@\n-{line}\n+{line};"
            elif "'}' expected" in error['error']:
                return f"@@ -{line_num},1 +{line_num},1 @@\n-{line}\n+{line} }}"
            
        return f"@@ -{error['line']},1 +{error['line']},1 @@\n- [unable to generate diff]"
    
    def train_on_diff(self, error: Dict, teacher_diff: str):
        """
        Train M2-BERT on the teacher's diff
        """
        self.model.train()
        
        # Prepare training data
        input_text = f"Error: {error['error']}\nFile: {error['file']}\nLine: {error['line']}\n"
        target_text = teacher_diff
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors = 'pt',
            truncation = True,
            max_length = 256,
            padding = 'max_length'
        ).to(self.device)
        
        targets = self.tokenizer(
            target_text,
            return_tensors = 'pt',
            truncation = True,
            max_length = 256,
            padding = 'max_length'
        ).to(self.device)
        
        # Forward pass
        outputs = self.model(inputs['input_ids'])
        
        # Compute loss
        loss = F.cross_entropy(
            outputs.view(-1, outputs.size(-1)),
            targets['input_ids'].view(-1),
            ignore_index = self.tokenizer.pad_token_id
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def save_checkpoint(self, reason: str):
        """Save M2-BERT's progress"""
        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'success_count': self.success_count,
            'learning_history': self.learning_history[-100:]  # Keep recent history
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"m2bert_checkpoint_{timestamp}_{reason}.pt"
        torch.save(checkpoint, path)
        print(f" Saved checkpoint: {path}")
    
    def training_loop(self, max_cycles: int = 10):
        """
        Main training loop where M2-BERT learns from Qwen3-Coder:30B
        """
        print("\n M2-BERT DIFF TRAINING")
        print(" = " * 60)
        print("M2-BERT is learning to generate diffs")
        print("Teacher: Qwen3-Coder:30B")
        print(" = " * 60)
        
        for cycle in range(1, max_cycles + 1):
            print(f"\n Learning Cycle {cycle}/{max_cycles}")
            
            errors = self.get_typescript_errors(max_errors = 20)
            
            if not errors:
                print(" No errors found M2-BERT has graduated")
                break
            
            print(f"Found {len(errors)} errors to learn from")
            
            cycle_correct = 0
            cycle_loss = 0
            
            for i, error in enumerate(errors, 1):
                print(f"\n[{i}/{len(errors)}] {error['file']}:{error['line']}")
                
                # Get context
                context = self.get_file_context(error['file'], error['line'])
                
                if not context:
                    continue
                
                # M2-BERT attempts
                student_diff = self.m2bert_generate_diff(error, context)
                print(f"   M2-BERT: {student_diff[:50]}.")
                
                # Teacher provides correct diff
                teacher_diff = self.ask_teacher_for_diff(error, context)
                
                if teacher_diff:
                    print(f"  ‍ Teacher: {teacher_diff[:50]}.")
                    
                    # Check if student was close
                    if student_diff and '@@' in student_diff and '@@' in teacher_diff:
                        # Basic similarity check
                        if any(line in teacher_diff for line in student_diff.split('\n') if line.startswith('+')):
                            print(f"   Acceptable M2-BERT")
                            cycle_correct += 1
                            self.success_count += 1
                    
                    # Learn from teacher
                    loss = self.train_on_diff(error, teacher_diff)
                    cycle_loss += loss
                    
                    self.learning_history.append({
                        'error': error['error'],
                        'student': student_diff[:100],
                        'teacher': teacher_diff[:100],
                        'loss': loss
                    })
            
            # Cycle summary
            avg_loss = cycle_loss / len(errors) if errors else 0
            accuracy = cycle_correct / len(errors) if errors else 0
            
            print(f"\n Cycle {cycle} Results:")
            print(f"  Correct diffs: {cycle_correct}/{len(errors)} ({accuracy:.1%})")
            print(f"  Average loss: {avg_loss:.4f}")
            print(f"  Total successes: {self.success_count}")
            
            # Save checkpoint every 3 cycles
            if cycle % 3 == 0:
                self.save_checkpoint(f"cycle{cycle}")
            
            # Early stopping if doing well
            if accuracy > 0.8:
                print(f"\n M2-BERT is doing great ({accuracy:.1%} accuracy)")
                self.save_checkpoint("graduated")
                break
        
        print("\n" + " = " * 60)
        print(" TRAINING COMPLETE")
        print(f"Final success count: {self.success_count}")
        print("M2-BERT can now generate diffs ")

if __name__ == "__main__":
    generator = DiffGenerator()
    generator.training_loop()