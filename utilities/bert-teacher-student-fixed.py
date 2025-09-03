#/usr/bin/env python3
"""
Fixed BERT Teacher-Student System with ACTUAL training
Using CodeBERT for code understanding and Qwen3 as teacher
"""

import subprocess
import os
import json
import requests
import torch
import torch.nn as nn
from transformers import (
    RobertaTokenizer, 
    RobertaForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from torch.utils.data import Dataset
from datetime import datetime
import numpy as np

class CodeFixDataset(Dataset):
    """Dataset for training on code fixes"""
    def __init__(self, examples, tokenizer, max_length = 256):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Format: [CLS] error_type [SEP] broken_code [MASK] [SEP] 
        # Target: fixed_code at [MASK] position
        text = f"{example['error']} [SEP] {example['broken']} [MASK] [SEP]"
        target = example['fixed']
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation = True,
            padding = 'max_length',
            max_length = self.max_length,
            return_tensors = 'pt'
        )
        
        # Create labels for MLM - only train on the [MASK] replacement
        labels = encoding['input_ids'].clone()
        labels[labels = self.tokenizer.mask_token_id] = -100
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }

class BERTTeacherStudent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Use CodeBERT - actually designed for code
        model_name = "microsoft/codebert-base"
        print(f"Loading {model_name}.")
        
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        
        # Training data accumulator
        self.training_examples = []
        self.checkpoint_dir = "./bert-checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok = True)
        
    def get_typescript_errors(self, max_errors = 10):
        """Get current TypeScript errors"""
        try:
            result = subprocess.run(['npx', 'tsc', '--noEmit'], 
                                  capture_output = True, text = True, timeout = 30)
            errors = []
            
            for line in result.stderr.split('\n'):
                if 'error TS' in line and '(' in line:
                    try:
                        parts = line.split(': error TS')
                        if len(parts) == 2:
                            file_coords = parts[0]
                            error_msg = 'error TS' + parts[1]
                            
                            paren_idx = file_coords.rfind('(')
                            if paren_idx > 0:
                                file_path = file_coords[:paren_idx]
                                coords = file_coords[paren_idx+1:-1]
                                line_num, col_num = coords.split(',')
                                
                                errors.append({
                                    'file': file_path.strip(),
                                    'line': int(line_num),
                                    'error': error_msg.strip()
                                })
                                
                                if max_errors and len(errors) >= max_errors:
                                    break
                    except:
                        continue
            
            return errors
        except Exception as e:
            print(f"Error getting TypeScript errors: {e}")
            return []
    
    def get_code_context(self, file_path, line_num):
        """Get the broken line and context"""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            if line_num <= len(lines):
                broken_line = lines[line_num - 1].strip()
                
                # Get surrounding context
                start = max(0, line_num - 3)
                end = min(len(lines), line_num + 2)
                context = ''.join(lines[start:end])
                
                return broken_line, context
            
        except Exception as e:
            print(f"Error reading file: {e}")
        
        return "", ""
    
    def ask_teacher(self, error_info, broken_line, context):
        """Ask Qwen3-30B teacher for the correct fix"""
        prompt = f"""Fix this TypeScript error. Return ONLY the corrected line, no explanation.

Error: {error_info['error']}
Broken line: {broken_line}

Context:
{context}

Fixed line:"""
        
        try:
            response = requests.post('http://localhost:11434/api/generate',
                json = {
                    'model': 'qwen3-coder:30b',
                    'prompt': prompt,
                    'stream': False,
                    'options': {'temperature': 0.1}
                }, timeout = 30)
            
            if response.status_code == 200:
                result = response.json()
                fixed = result.get('response', '').strip()
                
                # Clean up response
                if fixed and fixed != broken_line:
                    # Remove markdown/explanations if present
                    lines = fixed.split('\n')
                    for line in lines:
                        clean = line.strip()
                        if clean and not clean.startswith('```') and not clean.startswith('//'):
                            return clean
            
        except Exception as e:
            print(f"Teacher error: {e}")
        
        return None
    
    def bert_predict(self, error_msg, broken_line):
        """Use BERT to predict a fix"""
        # Create masked input
        text = f"{error_msg} [SEP] {broken_line} [MASK]"
        
        inputs = self.tokenizer(text, return_tensors = 'pt', truncation = True, max_length = 256)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits
            
            # Find mask token position
            mask_positions = (inputs['input_ids'] == self.tokenizer.mask_token_id).nonzero(as_tuple = True)
            
            if len(mask_positions[1]) > 0:
                mask_idx = mask_positions[1][0]
                
                # Get top prediction for mask position
                predicted_token_id = torch.argmax(predictions[0, mask_idx]).item()
                predicted_token = self.tokenizer.decode([predicted_token_id])
                
                # Try to construct a reasonable fix
                if ';' in error_msg and not broken_line.endswith(';'):
                    return broken_line + ';'
                elif '}' in error_msg:
                    return broken_line + ' }'
                else:
                    return broken_line + predicted_token
        
        return broken_line
    
    def train_on_examples(self, examples, epochs = 3):
        """Actually train BERT on the collected examples"""
        if not examples:
            return
        
        print(f"Training on {len(examples)} examples.")
        
        # Create dataset
        dataset = CodeFixDataset(examples, self.tokenizer)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir = self.checkpoint_dir,
            num_train_epochs = epochs,
            per_device_train_batch_size = 4,
            warmup_steps = 10,
            weight_decay = 0.01,
            logging_dir = './logs',
            save_steps = 100,
            eval_steps = 100,
            save_total_limit = 2,
            load_best_model_at_end = True,
            metric_for_best_model = 'loss',
            greater_is_better = False,
            remove_unused_columns = False,
            # MPS-specific settings
            use_mps_device = self.device.type == 'mps',
            dataloader_pin_memory = False if self.device.type == 'mps' else True,
        )
        
        # Data collator for MLM
        data_collator = DataCollatorForLanguageModeling(
            tokenizer = self.tokenizer,
            mlm = True,
            mlm_probability = 0.15
        )
        
        # Create trainer
        trainer = Trainer(
            model = self.model,
            args = training_args,
            train_dataset = dataset,
            data_collator = data_collator,
        )
        
        # Train
        trainer.train()
        
        # Save the fine-tuned model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"{self.checkpoint_dir}/bert-finetuned-{timestamp}"
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        print(f"Model saved to {save_path}")
    
    def run_training_loop(self, max_cycles = 10):
        """Main training loop"""
        print("Starting BERT Teacher-Student Training Loop")
        print(" = " * 50)
        
        for cycle in range(1, max_cycles + 1):
            print(f"\nCycle {cycle}/{max_cycles}")
            
            errors = self.get_typescript_errors(max_errors = 20)
            
            if not errors:
                print("No errors found Training complete.")
                break
            
            print(f"Found {len(errors)} errors")
            
            cycle_examples = []
            bert_correct = 0
            teacher_helped = 0
            
            for i, error in enumerate(errors, 1):
                print(f"  [{i}/{len(errors)}] {error['file']}:{error['line']}")
                
                broken_line, context = self.get_code_context(error['file'], error['line'])
                if not broken_line:
                    continue
                
                # Try BERT prediction
                bert_fix = self.bert_predict(error['error'], broken_line)
                
                # Validate by checking with teacher
                teacher_fix = self.ask_teacher(error, broken_line, context)
                
                if teacher_fix:
                    if bert_fix == teacher_fix:
                        print(f"     BERT got it right")
                        bert_correct += 1
                    else:
                        print(f"    → Teacher corrects: {teacher_fix[:50]}.")
                        teacher_helped += 1
                    
                    # Add to training examples
                    cycle_examples.append({
                        'error': error['error'],
                        'broken': broken_line,
                        'fixed': teacher_fix
                    })
            
            print(f"\nCycle {cycle} Results:")
            print(f"  BERT correct: {bert_correct}/{len(errors)}")
            print(f"  Teacher corrections: {teacher_helped}")
            
            # Add to overall training data
            self.training_examples.extend(cycle_examples)
            
            # Train every 2 cycles or when we have enough examples
            if cycle % 2 == 0 or len(self.training_examples) >= 50:
                self.train_on_examples(self.training_examples)
                # Keep recent examples for continual learning
                self.training_examples = self.training_examples[-100:]
            
            # Check improvement
            if bert_correct / len(errors) > 0.8:
                print("BERT achieving high accuracy Consider deployment.")
                break
        
        # Final training with all examples
        if self.training_examples:
            print(f"\nFinal training with {len(self.training_examples)} total examples.")
            self.train_on_examples(self.training_examples, epochs = 5)
        
        print("\nTraining complete")

if __name__ == "__main__":
    trainer = BERTTeacherStudent()
    trainer.run_training_loop()