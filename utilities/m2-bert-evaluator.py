#!/usr/bin/env python3
"""
M2-BERT Evaluator
Based on the actual evaluation methodology from Poli et al.'s Monarch Mixer work

Key metrics:
1. MLM Loss (for BERT-style models, not perplexity which is for autoregressive)
2. GLUE benchmark scores (for downstream task evaluation)
3. Training efficiency metrics (samples/sec, memory usage)
4. Numerical stability monitoring

Note: Perplexity is NOT appropriate for BERT-style masked models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path
import json
import time
from dataclasses import dataclass
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

@dataclass
class TrainingMetrics:
    """Metrics tracked during M2-BERT training"""
    step: int
    mlm_loss: float
    mlm_accuracy: float
    gradient_norm: float
    learning_rate: float
    samples_per_second: float
    memory_gb: float
    timestamp: float

class M2BertEvaluator:
    """
    Evaluator for M2-BERT training based on Poli et al.'s methodology
    
    Critical details from the paper:
    - MLM probability: 0.30 for training, 0.15 for evaluation
    - Eval interval: 2000 batches
    - Batch size: 4096
    - Gradient clipping norm: 1.0
    - Target GLUE scores: 79.9 (80M), 80.9 (110M)
    """
    
    def __init__(self, 
                 model_dim: int = 960,
                 mlm_prob_train: float = 0.30,
                 mlm_prob_eval: float = 0.15,
                 eval_interval: int = 2000,
                 checkpoint_interval: int = 7000):
        """
        Initialize evaluator with M2-BERT hyperparameters
        
        Args:
            model_dim: Hidden dimension (768 for 80M, 960 for 110M)
            mlm_prob_train: MLM probability during training
            mlm_prob_eval: MLM probability during evaluation
            eval_interval: Evaluate every N batches
            checkpoint_interval: Save checkpoint every N batches
        """
        self.model_dim = model_dim
        self.mlm_prob_train = mlm_prob_train
        self.mlm_prob_eval = mlm_prob_eval
        self.eval_interval = eval_interval
        self.checkpoint_interval = checkpoint_interval
        
        # Track metrics
        self.training_metrics = []
        self.validation_metrics = []
        
        # Expected performance targets
        self.glue_targets = {
            80: 79.9,   # 80M model
            110: 80.9,  # 110M model
            260: 82.2,  # 260M model
            341: 82.8   # 341M model
        }
        
        # Numerical stability thresholds
        self.gradient_clip_norm = 1.0
        self.max_gradient_norm = 10.0  # Warning threshold
        self.min_loss = 0.1  # Suspiciously low
        self.max_loss = 10.0  # Divergence warning
        
    def compute_mlm_loss(self, 
                        logits: torch.Tensor,
                        labels: torch.Tensor,
                        attention_mask: Optional[torch.Tensor] = None) -> Tuple[float, float]:
        """
        Compute MLM loss and accuracy
        
        This is the PRIMARY metric for BERT-style models (NOT perplexity!)
        """
        # Only compute loss on masked tokens
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Reshape for loss computation
        vocab_size = logits.size(-1)
        mlm_loss = loss_fct(
            logits.view(-1, vocab_size),
            labels.view(-1)
        )
        
        # Compute accuracy on masked tokens
        masked_tokens = labels != -100
        if masked_tokens.sum() > 0:
            predictions = logits.argmax(dim=-1)
            correct = (predictions == labels) & masked_tokens
            accuracy = correct.sum().float() / masked_tokens.sum().float()
        else:
            accuracy = 0.0
            
        return mlm_loss.item(), accuracy.item()
    
    def check_gradient_health(self, model: nn.Module) -> Dict[str, float]:
        """
        Monitor gradient norms for numerical stability
        
        Based on M2-BERT config: gradient clipping norm = 1.0
        """
        total_norm = 0.0
        param_count = 0
        min_grad = float('inf')
        max_grad = float('-inf')
        
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                param_count += 1
                min_grad = min(min_grad, param_norm)
                max_grad = max(max_grad, param_norm)
        
        total_norm = total_norm ** 0.5
        
        health = {
            'total_norm': total_norm,
            'avg_norm': total_norm / max(param_count, 1),
            'min_grad': min_grad,
            'max_grad': max_grad,
            'requires_clipping': total_norm > self.gradient_clip_norm
        }
        
        # Warnings
        if total_norm > self.max_gradient_norm:
            print(f"⚠️  WARNING: Gradient norm {total_norm:.2f} exceeds threshold {self.max_gradient_norm}")
        
        if total_norm < 1e-6:
            print(f"⚠️  WARNING: Gradient norm {total_norm:.2e} is suspiciously small")
            
        return health
    
    def evaluate_step(self,
                     model: nn.Module,
                     eval_dataloader,
                     step: int) -> Dict[str, float]:
        """
        Run evaluation at specified interval
        
        Matches M2-BERT eval protocol:
        - MLM probability: 0.15 for evaluation
        - No gradient computation
        """
        model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                if num_batches >= 10:  # Quick eval
                    break
                    
                # Forward pass
                outputs = model(batch['input_ids'])
                
                # Compute metrics
                loss, accuracy = self.compute_mlm_loss(
                    outputs,
                    batch['labels']
                )
                
                total_loss += loss
                total_accuracy += accuracy
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_accuracy = total_accuracy / max(num_batches, 1)
        
        # Check for concerning values
        if avg_loss < self.min_loss:
            print(f"⚠️  Suspiciously low loss: {avg_loss:.4f}")
        elif avg_loss > self.max_loss:
            print(f"⚠️  Loss diverging: {avg_loss:.4f}")
            
        return {
            'eval_loss': avg_loss,
            'eval_accuracy': avg_accuracy,
            'step': step
        }
    
    def should_evaluate(self, step: int) -> bool:
        """Check if we should run evaluation"""
        return step % self.eval_interval == 0
    
    def should_checkpoint(self, step: int) -> bool:
        """Check if we should save checkpoint"""
        return step % self.checkpoint_interval == 0
    
    def track_training_step(self,
                           step: int,
                           loss: float,
                           accuracy: float,
                           gradient_norm: float,
                           learning_rate: float,
                           batch_time: float,
                           batch_size: int) -> TrainingMetrics:
        """
        Track metrics for each training step
        """
        # Calculate throughput
        samples_per_second = batch_size / batch_time
        
        # Estimate memory (simplified)
        memory_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        
        metrics = TrainingMetrics(
            step=step,
            mlm_loss=loss,
            mlm_accuracy=accuracy,
            gradient_norm=gradient_norm,
            learning_rate=learning_rate,
            samples_per_second=samples_per_second,
            memory_gb=memory_gb,
            timestamp=time.time()
        )
        
        self.training_metrics.append(metrics)
        
        return metrics
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """
        Plot training curves similar to standard practice
        """
        if not self.training_metrics:
            print("No training metrics to plot")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        steps = [m.step for m in self.training_metrics]
        
        # Loss curve
        losses = [m.mlm_loss for m in self.training_metrics]
        axes[0, 0].plot(steps, losses)
        axes[0, 0].set_title('MLM Loss')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].axhline(y=2.0, color='g', linestyle='--', label='Target')
        
        # Accuracy curve
        accuracies = [m.mlm_accuracy for m in self.training_metrics]
        axes[0, 1].plot(steps, accuracies)
        axes[0, 1].set_title('MLM Accuracy')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].axhline(y=0.7, color='g', linestyle='--', label='Target')
        
        # Gradient norm
        grad_norms = [m.gradient_norm for m in self.training_metrics]
        axes[1, 0].plot(steps, grad_norms)
        axes[1, 0].set_title('Gradient Norm')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Norm')
        axes[1, 0].axhline(y=self.gradient_clip_norm, color='r', linestyle='--', label='Clip threshold')
        
        # Learning rate
        lrs = [m.learning_rate for m in self.training_metrics]
        axes[1, 1].plot(steps, lrs)
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('LR')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def validate_against_baseline(self, current_metrics: Dict[str, float], model_size: int = 110):
        """
        Compare current performance against M2-BERT baselines
        """
        if model_size in self.glue_targets:
            target = self.glue_targets[model_size]
            print(f"\nBaseline comparison for {model_size}M model:")
            print(f"  Target GLUE score: {target}")
            print(f"  Current MLM loss: {current_metrics.get('eval_loss', 'N/A'):.4f}")
            print(f"  Current MLM accuracy: {current_metrics.get('eval_accuracy', 'N/A'):.2%}")
            
            # Rough heuristic: MLM accuracy of 70%+ usually correlates with good GLUE
            if current_metrics.get('eval_accuracy', 0) > 0.7:
                print("  ✓ On track for good downstream performance")
            else:
                print("  ⚠️  May need more training for target GLUE score")
    
    def save_metrics(self, filepath: str):
        """Save all metrics to JSON"""
        data = {
            'training': [
                {
                    'step': m.step,
                    'loss': m.mlm_loss,
                    'accuracy': m.mlm_accuracy,
                    'gradient_norm': m.gradient_norm,
                    'lr': m.learning_rate,
                    'throughput': m.samples_per_second
                }
                for m in self.training_metrics
            ],
            'validation': self.validation_metrics,
            'config': {
                'model_dim': self.model_dim,
                'mlm_prob_train': self.mlm_prob_train,
                'mlm_prob_eval': self.mlm_prob_eval,
                'eval_interval': self.eval_interval
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
    def print_summary(self):
        """Print training summary"""
        if not self.training_metrics:
            print("No metrics collected")
            return
            
        latest = self.training_metrics[-1]
        
        print("\n" + "="*60)
        print("M2-BERT TRAINING SUMMARY")
        print("="*60)
        print(f"Total steps: {latest.step}")
        print(f"Final MLM loss: {latest.mlm_loss:.4f}")
        print(f"Final MLM accuracy: {latest.mlm_accuracy:.2%}")
        print(f"Final gradient norm: {latest.gradient_norm:.2f}")
        print(f"Throughput: {latest.samples_per_second:.1f} samples/sec")
        
        # Averages
        avg_loss = np.mean([m.mlm_loss for m in self.training_metrics[-100:]])
        avg_acc = np.mean([m.mlm_accuracy for m in self.training_metrics[-100:]])
        
        print(f"\nLast 100 steps:")
        print(f"  Avg MLM loss: {avg_loss:.4f}")
        print(f"  Avg MLM accuracy: {avg_acc:.2%}")
        
        # Stability check
        loss_std = np.std([m.mlm_loss for m in self.training_metrics[-100:]])
        if loss_std > 0.5:
            print(f"  ⚠️  High loss variance: {loss_std:.2f}")
        else:
            print(f"  ✓ Stable training (loss std: {loss_std:.3f})")

if __name__ == "__main__":
    # Demo evaluation
    evaluator = M2BertEvaluator()
    
    print("M2-BERT Evaluator initialized")
    print(f"Config:")
    print(f"  MLM prob (train): {evaluator.mlm_prob_train}")
    print(f"  MLM prob (eval): {evaluator.mlm_prob_eval}")
    print(f"  Eval interval: {evaluator.eval_interval} batches")
    print(f"  Checkpoint interval: {evaluator.checkpoint_interval} batches")
    print(f"\nTarget GLUE scores:")
    for size, score in evaluator.glue_targets.items():
        print(f"  {size}M model: {score}")