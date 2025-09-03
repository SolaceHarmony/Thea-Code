#/usr/bin/env python3
"""
M2-BERT Continuous Learning System
When diffs fail validation, M2-BERT learns and improves
Stores all training data as JSON for reproducibility
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
from typing import Dict, List, Optional
import hashlib
from pathlib import Path

class M2BertModel(nn.Module):
    """Our continuously learning M2-BERT"""
    def __init__(self, vocab_size = 8000, dim = 256, max_seq_len = 512, n_layers = 4):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, dim) * 0.02)
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model = dim,
                nhead = 8,
                dim_feedforward = dim * 2,
                dropout = 0.1,
                batch_first = True
            ) for _ in range(n_layers)
        ])
        
        self.diff_head = nn.Linear(dim, vocab_size)
        
    def forward(self, input_ids, attention_mask = None):
        seq_len = input_ids.size(1)
        x = self.embedding(input_ids)
        x = x + self.pos_encoding[:, :seq_len, :]
        
        for layer in self.layers:
            x = layer(x)
        
        return self.diff_head(x)

class TrainingDataStore:
    """Manages JSON-based training data storage"""
    
    def __init__(self, data_dir = "m2bert_training_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok = True)
        
        self.current_session_file = self.data_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.master_file = self.data_dir / "master_training_data.json"
        
        self.session_data = []
        self.load_master_data()
    
    def load_master_data(self):
        """Load accumulated training data"""
        if self.master_file.exists():
            with open(self.master_file, 'r') as f:
                self.master_data = json.load(f)
        else:
            self.master_data = {
                'total_examples': 0,
                'successful_fixes': 0,
                'failed_attempts': 0,
                'examples': []
            }
    
    def save_training_example(self, example: Dict):
        """Save a training example"""
        # Add metadata
        example['timestamp'] = datetime.now().isoformat()
        example['example_id'] = hashlib.md5(
            f"{example['error']}{example['timestamp']}".encode()
        ).hexdigest()[:8]
        
        # Add to session data
        self.session_data.append(example)
        
        # Save session file
        with open(self.current_session_file, 'w') as f:
            json.dump(self.session_data, f, indent = 2)
        
        # Update master data
        self.master_data['examples'].append(example)
        self.master_data['total_examples'] += 1
        
        if example.get('validation_passed', False):
            self.master_data['successful_fixes'] += 1
        else:
            self.master_data['failed_attempts'] += 1
        
        # Save master file (keep only last 1000 examples to prevent bloat)
        self.master_data['examples'] = self.master_data['examples'][-1000:]
        
        with open(self.master_file, 'w') as f:
            json.dump(self.master_data, f, indent = 2)
    
    def get_training_batch(self, batch_size = 8) -> List[Dict]:
        """Get a batch of training examples for replay training"""
        if len(self.master_data['examples']) < batch_size:
            return self.master_data['examples']
        
        import random
        return random.sample(self.master_data['examples'], batch_size)

class ContinuousLearner:
    """Complete continuous learning system with validation and refinement"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                  "mps" if torch.backends.mps.is_available() else "cpu")
        
        print(f" M2-BERT Continuous Learner on {self.device}")
        
        # Initialize components
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = M2BertModel(vocab_size = self.tokenizer.vocab_size).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = 5e-4)
        
        # Training data storage
        self.data_store = TrainingDataStore()
        
        # Checkpoint management
        self.checkpoint_dir = Path("m2bert_checkpoints")
        self.checkpoint_dir.mkdir(exist_ok = True)
        self.best_accuracy = 0.0
        self.generation_count = 0
        
    def generate_diff(self, error: Dict, context: str) -> str:
        """Generate a diff using M2-BERT"""
        self.model.eval()
        self.generation_count += 1
        
        # Format input
        input_text = json.dumps({
            'error': error['error'],
            'file': error['file'],
            'line': error['line'],
            'context': context[:200]
        })
        
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
            generated = self.tokenizer.decode(predictions[0], skip_special_tokens = True)
        
        # Try to extract diff pattern
        if '@@' in generated:
            return generated
        
        # Fallback to simple diff generation
        return self.generate_simple_diff(error, context)
    
    def generate_simple_diff(self, error: Dict, context: str) -> str:
        """Generate a simple diff based on error type"""
        line_num = error['line']
        
        if "';' expected" in error['error']:
            return f"@@ -{line_num},1 +{line_num},1 @@\n-{context}\n+{context};"
        elif "'}' expected" in error['error']:
            return f"@@ -{line_num},1 +{line_num},1 @@\n-{context}\n+{context} }}"
        else:
            return f"@@ -{line_num},1 +{line_num},1 @@\n-{context}\n+{context} // TODO: fix"
    
    def validate_diff(self, diff: str, file_path: str) -> Dict:
        """Validate diff with linter and Qwen3"""
        validation_result = {
            'lint_passed': False,
            'qwen_approved': False,
            'validation_passed': False,
            'feedback': []
        }
        
        # Quick lint check (simplified for demo)
        try:
            result = subprocess.run(
                ['npx', 'tsc', '--noEmit', file_path],
                capture_output = True,
                text = True,
                timeout = 10
            )
            validation_result['lint_passed'] = 'error' not in result.stderr.lower()
        except:
            pass
        
        # Ask Qwen for approval
        try:
            response = requests.post('http://localhost:11434/api/generate',
                json = {
                    'model': 'qwen3-coder:30b',
                    'prompt': f"Is this diff correct? Reply YES or NO with reason.\n\n{diff[:200]}\n\nAnswer:",
                    'stream': False,
                    'options': {'temperature': 0.1}
                }, timeout = 10)
            
            if response.status_code == 200:
                answer = response.json().get('response', '')
                validation_result['qwen_approved'] = 'yes' in answer.lower()
                validation_result['feedback'].append(answer[:100])
        except:
            pass
        
        validation_result['validation_passed'] = (
            validation_result['lint_passed'] and 
            validation_result['qwen_approved']
        )
        
        return validation_result
    
    def refine_training(self, error: Dict, bad_diff: str, correct_diff: str):
        """Train M2-BERT when its diff fails validation"""
        self.model.train()
        
        print(f"   Refinement training on failed attempt")
        
        # Prepare training data
        input_data = json.dumps({
            'error': error['error'],
            'file': error['file'],
            'line': error['line'],
            'bad_attempt': bad_diff[:100]
        })
        
        target_data = correct_diff
        
        # Tokenize
        inputs = self.tokenizer(
            input_data,
            return_tensors = 'pt',
            truncation = True,
            max_length = 256,
            padding = 'max_length'
        ).to(self.device)
        
        targets = self.tokenizer(
            target_data,
            return_tensors = 'pt',
            truncation = True,
            max_length = 256,
            padding = 'max_length'
        ).to(self.device)
        
        # Multiple training iterations for failed attempts
        for epoch in range(3):
            outputs = self.model(inputs['input_ids'])
            
            loss = F.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                targets['input_ids'].view(-1),
                ignore_index = self.tokenizer.pad_token_id
            )
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            print(f"    Epoch {epoch+1}/3 - Loss: {loss.item():.4f}")
        
        return loss.item()
    
    def replay_training(self):
        """Periodically replay past examples to prevent forgetting"""
        batch = self.data_store.get_training_batch(batch_size = 8)
        
        if not batch:
            return
        
        print(f"\n Replay training on {len(batch)} past examples")
        
        self.model.train()
        total_loss = 0
        
        for example in batch:
            if 'correct_diff' not in example:
                continue
            
            input_data = json.dumps({
                'error': example['error'],
                'context': example.get('context', '')[:200]
            })
            
            inputs = self.tokenizer(
                input_data,
                return_tensors = 'pt',
                truncation = True,
                max_length = 256,
                padding = 'max_length'
            ).to(self.device)
            
            targets = self.tokenizer(
                example['correct_diff'],
                return_tensors = 'pt',
                truncation = True,
                max_length = 256,
                padding = 'max_length'
            ).to(self.device)
            
            outputs = self.model(inputs['input_ids'])
            loss = F.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                targets['input_ids'].view(-1),
                ignore_index = self.tokenizer.pad_token_id
            )
            
            loss.backward()
            total_loss += loss.item()
        
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        avg_loss = total_loss / len(batch)
        print(f"  Replay training loss: {avg_loss:.4f}")
    
    def save_checkpoint(self, accuracy: float, reason: str = "periodic"):
        """Save model checkpoint with metadata"""
        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'accuracy': accuracy,
            'generation_count': self.generation_count,
            'training_stats': self.data_store.master_data,
            'timestamp': datetime.now().isoformat()
        }
        
        filename = f"checkpoint_{reason}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        path = self.checkpoint_dir / filename
        
        torch.save(checkpoint, path)
        print(f" Checkpoint saved: {filename} (accuracy: {accuracy:.2%})")
        
        # Save best model
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f" New best model Accuracy: {accuracy:.2%}")
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None):
        """Load a saved checkpoint"""
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / "best_model.pt"
        
        if not Path(checkpoint_path).exists():
            print("No checkpoint found")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location = self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.generation_count = checkpoint.get('generation_count', 0)
        
        print(f" Loaded checkpoint: {checkpoint_path}")
        print(f"  Previous accuracy: {checkpoint.get('accuracy', 0):.2%}")
        print(f"  Generation count: {self.generation_count}")
    
    def continuous_learning_loop(self, max_iterations = 100):
        """Main continuous learning loop"""
        print("\n CONTINUOUS LEARNING SYSTEM")
        print(" = "*60)
        print("M2-BERT learns from every attempt, successful or not")
        print(" = "*60)
        
        # Try to load previous best model
        self.load_checkpoint()
        
        successful_fixes = 0
        total_attempts = 0
        
        for iteration in range(1, max_iterations + 1):
            print(f"\n Learning Iteration {iteration}/{max_iterations}")
            
            # Get TypeScript errors
            errors = self.get_typescript_errors(max_errors = 10)
            
            if not errors:
                print(" No errors found")
                break
            
            iteration_success = 0
            
            for error in errors:
                total_attempts += 1
                
                print(f"\n  Attempt {total_attempts}: {error['file']}:{error['line']}")
                
                # Get context
                context = self.get_file_context(error['file'], error['line'])
                
                # Generate diff
                m2bert_diff = self.generate_diff(error, context)
                print(f"   M2-BERT generated diff")
                
                # Validate
                validation = self.validate_diff(m2bert_diff, error['file'])
                
                training_example = {
                    'error': error['error'],
                    'file': error['file'],
                    'line': error['line'],
                    'context': context,
                    'm2bert_diff': m2bert_diff,
                    'validation': validation,
                    'validation_passed': validation['validation_passed']
                }
                
                if validation['validation_passed']:
                    print(f"   Validation passed")
                    successful_fixes += 1
                    iteration_success += 1
                    
                    # Save successful example
                    training_example['correct_diff'] = m2bert_diff
                    self.data_store.save_training_example(training_example)
                    
                else:
                    print(f"   Validation failed: {validation['feedback']}")
                    
                    # Get correct diff from teacher
                    correct_diff = self.get_teacher_diff(error, context)
                    
                    if correct_diff:
                        # Refine training
                        loss = self.refine_training(error, m2bert_diff, correct_diff)
                        
                        # Save failed example with correction
                        training_example['correct_diff'] = correct_diff
                        training_example['refinement_loss'] = loss
                        self.data_store.save_training_example(training_example)
            
            # Calculate iteration accuracy
            iteration_accuracy = iteration_success / len(errors) if errors else 0
            overall_accuracy = successful_fixes / total_attempts if total_attempts else 0
            
            print(f"\n Iteration {iteration} Results:")
            print(f"  This iteration: {iteration_success}/{len(errors)} ({iteration_accuracy:.1%})")
            print(f"  Overall: {successful_fixes}/{total_attempts} ({overall_accuracy:.1%})")
            
            # Periodic replay training
            if iteration % 5 == 0:
                self.replay_training()
            
            # Save checkpoint
            if iteration % 10 == 0 or iteration_accuracy > 0.7:
                self.save_checkpoint(overall_accuracy, "periodic" if iteration % 10 == 0 else "high_accuracy")
            
            # Early stopping if doing well
            if iteration_accuracy >= 0.9:
                print(f"\n M2-BERT achieving {iteration_accuracy:.1%} accuracy")
                self.save_checkpoint(overall_accuracy, "excellent")
                
                if iteration > 20:  # Give it time to learn first
                    break
        
        # Final summary
        print("\n" + " = "*60)
        print("CONTINUOUS LEARNING COMPLETE")
        print(" = "*60)
        print(f"Total attempts: {total_attempts}")
        print(f"Successful fixes: {successful_fixes}")
        print(f"Final accuracy: {successful_fixes/total_attempts:.1%}" if total_attempts else "N/A")
        print(f"Training data saved: {len(self.data_store.session_data)} examples")
        print(f"Best model accuracy: {self.best_accuracy:.2%}")
        
        # Save final checkpoint
        self.save_checkpoint(successful_fixes/total_attempts if total_attempts else 0, "final")
    
    def get_typescript_errors(self, max_errors = 10) -> List[Dict]:
        """Get TypeScript compilation errors"""
        try:
            result = subprocess.run(['npx', 'tsc', '--noEmit'], 
                                  capture_output = True, text = True, timeout = 30)
            errors = []
            
            for line in result.stderr.split('\n')[:max_errors]:
                if 'error TS' in line and '(' in line:
                    try:
                        parts = line.split(': error TS')
                        if len(parts) == 2:
                            file_coords = parts[0]
                            error_msg = 'error TS' + parts[1]
                            
                            paren_idx = file_coords.rfind('(')
                            if paren_idx > 0:
                                file_path = file_coords[:paren_idx].strip()
                                coords = file_coords[paren_idx+1:-1]
                                line_num = int(coords.split(',')[0])
                                
                                errors.append({
                                    'file': file_path,
                                    'line': line_num,
                                    'error': error_msg.strip()
                                })
                    except:
                        continue
            
            return errors
        except:
            return []
    
    def get_file_context(self, file_path: str, line_num: int) -> str:
        """Get the error line from file"""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                if line_num <= len(lines):
                    return lines[line_num - 1].rstrip()
        except:
            pass
        return ""
    
    def get_teacher_diff(self, error: Dict, context: str) -> Optional[str]:
        """Get correct diff from Qwen3 teacher"""
        try:
            prompt = f"""Generate a minimal diff to fix this TypeScript error.
Return ONLY the diff in unified format.

Error: {error['error']}
Line {error['line']}: {context}

Diff:"""
            
            response = requests.post('http://localhost:11434/api/generate',
                json = {
                    'model': 'qwen3-coder:30b',
                    'prompt': prompt,
                    'stream': False,
                    'options': {'temperature': 0.1}
                }, timeout = 20)
            
            if response.status_code == 200:
                diff = response.json().get('response', '').strip()
                if '@@' in diff or '-' in diff or '+' in diff:
                    return diff
        except:
            pass
        
        return None

if __name__ == "__main__":
    learner = ContinuousLearner()
    learner.continuous_learning_loop()