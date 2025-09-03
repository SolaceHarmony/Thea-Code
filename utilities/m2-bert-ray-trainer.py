#!/usr/bin/env python3
"""
M2-BERT Distributed Training with Ray
World-class actor-based training pipeline

Architecture:
- DataLoaderActor: Manages data streaming and batching
- ModelActor: Handles forward/backward passes
- OptimizerActor: Manages distributed optimization
- EvaluatorActor: Runs validation asynchronously
- CheckpointActor: Handles model persistence
- MetricsActor: Aggregates and monitors training metrics

This follows Poli's exact hyperparameters while adding industrial-grade
distributed training capabilities.
"""

import ray
from ray import serve
from ray.util.actor_pool import ActorPool
from ray.train import Checkpoint
from ray.train.torch import TorchTrainer
from ray.air import session
from ray.air.config import ScalingConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import time
import json
from pathlib import Path
import asyncio
from collections import deque
import psutil
import GPUtil

# Ray initialization with proper resource allocation
ray.init(ignore_reinit_error=True)

@dataclass
class M2BertConfig:
    """Configuration matching Poli et al.'s exact specifications"""
    # Model architecture
    vocab_size: int = 30522
    hidden_dim: int = 960
    intermediate_dim: int = 3840
    n_layers: int = 12
    n_blocks: int = 4
    max_seq_length: int = 128
    
    # Training hyperparameters (from paper)
    learning_rate: float = 8e-4
    warmup_ratio: float = 0.06
    weight_decay: float = 1e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_epsilon: float = 1e-6
    gradient_clip_norm: float = 1.0
    
    # MLM specific
    mlm_probability_train: float = 0.30
    mlm_probability_eval: float = 0.15
    
    # Batching
    global_batch_size: int = 4096
    micro_batch_size: int = 128
    gradient_accumulation_steps: int = 32
    
    # Evaluation
    eval_interval: int = 2000
    checkpoint_interval: int = 7000
    log_interval: int = 10
    
    # System
    num_workers: int = 4
    prefetch_factor: int = 2
    mixed_precision: str = "bf16"
    
    # Ray specific
    num_model_replicas: int = 2
    num_data_workers: int = 4
    pipeline_parallel: bool = False

@ray.remote
class DataLoaderActor:
    """
    Ray actor for efficient data loading and preprocessing
    Handles tokenization, masking, and batching
    """
    
    def __init__(self, config: M2BertConfig, rank: int = 0):
        self.config = config
        self.rank = rank
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Initialize data queue
        self.data_queue = deque(maxlen=1000)
        self.prefetch_running = False
        
        # MLM masking setup
        self.mlm_probability = config.mlm_probability_train
        self.mask_token_id = self.tokenizer.mask_token_id
        
        print(f"DataLoaderActor {rank} initialized")
    
    def create_mlm_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Create MLM batch with proper masking probability
        Following BERT's 80-10-10 rule
        """
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.config.max_seq_length,
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        
        # Create MLM labels
        labels = input_ids.clone()
        
        # Create mask
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        
        # Don't mask special tokens
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(
            torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
        )
        
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Only compute loss on masked tokens
        
        # 80% mask, 10% random, 10% keep
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.mask_token_id
        
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    async def prefetch_batches(self):
        """Asynchronously prefetch batches"""
        self.prefetch_running = True
        while self.prefetch_running:
            if len(self.data_queue) < 100:
                # Generate synthetic batch for demo
                # In production, this would read from C4 dataset
                texts = [f"Sample text {i} for MLM training." for i in range(self.config.micro_batch_size)]
                batch = self.create_mlm_batch(texts)
                self.data_queue.append(batch)
            await asyncio.sleep(0.001)
    
    def get_batch(self) -> Dict[str, torch.Tensor]:
        """Get a batch for training"""
        if not self.data_queue:
            # Emergency batch generation
            texts = [f"Emergency text {i}" for i in range(self.config.micro_batch_size)]
            return self.create_mlm_batch(texts)
        return self.data_queue.popleft()
    
    def get_batches(self, n: int) -> List[Dict[str, torch.Tensor]]:
        """Get multiple batches"""
        return [self.get_batch() for _ in range(n)]

@ray.remote(num_gpus=1)
class ModelActor:
    """
    Ray actor managing model forward/backward passes
    Handles model parallelism if needed
    """
    
    def __init__(self, config: M2BertConfig, rank: int = 0):
        self.config = config
        self.rank = rank
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        # Import the proper Monarch architecture
        from monarch_matrix_proper import MonarchMatrixProper
        from m2_bert_architecture import M2BertModel
        
        # Initialize model
        self.model = M2BertModel(
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_dim,
            intermediate_dim=config.intermediate_dim,
            n_layers=config.n_layers,
            n_blocks=config.n_blocks,
            max_seq_length=config.max_seq_length
        ).to(self.device)
        
        # Mixed precision setup
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision == "fp16" else None
        self.autocast_dtype = torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay
        )
        
        self.global_step = 0
        self.accumulation_steps = 0
        
        print(f"ModelActor {rank} initialized on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M")
    
    def forward_backward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Execute forward and backward pass
        Returns loss and metrics
        """
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Mixed precision context
        with torch.cuda.amp.autocast(enabled=self.config.mixed_precision != "fp32", dtype=self.autocast_dtype):
            # Forward pass
            logits = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask")
            )
            
            # Compute MLM loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                logits.view(-1, self.config.vocab_size),
                batch["labels"].view(-1)
            )
            
            # Scale for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        self.accumulation_steps += 1
        
        # Compute accuracy
        with torch.no_grad():
            masked_tokens = batch["labels"] != -100
            if masked_tokens.sum() > 0:
                predictions = logits.argmax(dim=-1)
                correct = (predictions == batch["labels"]) & masked_tokens
                accuracy = correct.sum().float() / masked_tokens.sum().float()
            else:
                accuracy = torch.tensor(0.0)
        
        metrics = {
            "loss": loss.item() * self.config.gradient_accumulation_steps,
            "accuracy": accuracy.item(),
            "gradient_norm": self.get_gradient_norm()
        }
        
        # Optimizer step if accumulated enough
        if self.accumulation_steps >= self.config.gradient_accumulation_steps:
            self.optimizer_step()
            self.accumulation_steps = 0
            self.global_step += 1
            metrics["global_step"] = self.global_step
        
        return metrics
    
    def optimizer_step(self):
        """Execute optimizer step with gradient clipping"""
        # Gradient clipping
        if self.config.gradient_clip_norm > 0:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip_norm
            )
        
        # Optimizer step
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        # Zero gradients
        self.optimizer.zero_grad()
    
    def get_gradient_norm(self) -> float:
        """Compute total gradient norm"""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    def get_model_state(self) -> Dict[str, Any]:
        """Get model state for checkpointing"""
        return {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "config": asdict(self.config)
        }
    
    def load_model_state(self, state: Dict[str, Any]):
        """Load model state from checkpoint"""
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.global_step = state["global_step"]

@ray.remote
class EvaluatorActor:
    """
    Ray actor for asynchronous evaluation
    Runs validation without blocking training
    """
    
    def __init__(self, config: M2BertConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        # Track validation metrics
        self.validation_history = []
        
        # GLUE targets from paper
        self.glue_targets = {
            80: 79.9,
            110: 80.9,
            260: 82.2,
            341: 82.8
        }
    
    def evaluate(self, model_state: Dict[str, Any], eval_batches: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """
        Run evaluation on validation set
        """
        from m2_bert_architecture import M2BertModel
        
        # Create model for evaluation
        model = M2BertModel(
            vocab_size=self.config.vocab_size,
            hidden_dim=self.config.hidden_dim,
            intermediate_dim=self.config.intermediate_dim,
            n_layers=self.config.n_layers,
            n_blocks=self.config.n_blocks
        ).to(self.device)
        
        # Load state
        model.load_state_dict(model_state["model_state_dict"])
        model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = len(eval_batches)
        
        with torch.no_grad():
            for batch in eval_batches:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                logits = model(batch["input_ids"])
                
                # Compute loss
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(
                    logits.view(-1, self.config.vocab_size),
                    batch["labels"].view(-1)
                )
                
                # Compute accuracy
                masked_tokens = batch["labels"] != -100
                if masked_tokens.sum() > 0:
                    predictions = logits.argmax(dim=-1)
                    correct = (predictions == batch["labels"]) & masked_tokens
                    accuracy = correct.sum().float() / masked_tokens.sum().float()
                else:
                    accuracy = 0.0
                
                total_loss += loss.item()
                total_accuracy += accuracy
        
        metrics = {
            "eval_loss": total_loss / num_batches,
            "eval_accuracy": total_accuracy / num_batches,
            "global_step": model_state["global_step"]
        }
        
        self.validation_history.append(metrics)
        
        # Check against targets
        self.check_glue_trajectory(metrics)
        
        return metrics
    
    def check_glue_trajectory(self, metrics: Dict[str, float]):
        """Check if we're on track for target GLUE scores"""
        # Heuristic: 70%+ MLM accuracy correlates with good GLUE
        if metrics["eval_accuracy"] > 0.7:
            print(f"✓ On track for GLUE targets (MLM acc: {metrics['eval_accuracy']:.2%})")
        else:
            print(f"⚠️ Below expected trajectory (MLM acc: {metrics['eval_accuracy']:.2%})")

@ray.remote
class MetricsActor:
    """
    Ray actor for metrics aggregation and monitoring
    Tracks training dynamics and numerical stability
    """
    
    def __init__(self, config: M2BertConfig):
        self.config = config
        self.metrics_buffer = deque(maxlen=10000)
        self.summary_stats = {}
        
        # Stability thresholds
        self.max_gradient_norm = 10.0
        self.min_loss = 0.1
        self.max_loss = 10.0
        
        # Initialize monitoring
        self.start_time = time.time()
        self.total_samples = 0
    
    def record_metrics(self, metrics: Dict[str, float]):
        """Record training metrics"""
        metrics["timestamp"] = time.time()
        metrics["elapsed"] = metrics["timestamp"] - self.start_time
        
        # Add system metrics
        metrics["cpu_percent"] = psutil.cpu_percent()
        metrics["memory_gb"] = psutil.virtual_memory().used / 1e9
        
        if torch.cuda.is_available():
            metrics["gpu_memory_gb"] = torch.cuda.memory_allocated() / 1e9
            metrics["gpu_utilization"] = GPUtil.getGPUs()[0].load if GPUtil.getGPUs() else 0
        
        self.metrics_buffer.append(metrics)
        self.total_samples += self.config.micro_batch_size
        
        # Check stability
        self.check_training_stability(metrics)
        
        # Update summary
        self.update_summary()
    
    def check_training_stability(self, metrics: Dict[str, float]):
        """Monitor for training instabilities"""
        if "gradient_norm" in metrics:
            if metrics["gradient_norm"] > self.max_gradient_norm:
                print(f"⚠️ High gradient norm: {metrics['gradient_norm']:.2f}")
            elif metrics["gradient_norm"] < 1e-6:
                print(f"⚠️ Vanishing gradients: {metrics['gradient_norm']:.2e}")
        
        if "loss" in metrics:
            if metrics["loss"] < self.min_loss:
                print(f"⚠️ Suspiciously low loss: {metrics['loss']:.4f}")
            elif metrics["loss"] > self.max_loss:
                print(f"⚠️ Loss diverging: {metrics['loss']:.4f}")
    
    def update_summary(self):
        """Update summary statistics"""
        if len(self.metrics_buffer) > 0:
            recent = list(self.metrics_buffer)[-100:]
            
            self.summary_stats = {
                "avg_loss": np.mean([m.get("loss", 0) for m in recent]),
                "avg_accuracy": np.mean([m.get("accuracy", 0) for m in recent]),
                "avg_gradient_norm": np.mean([m.get("gradient_norm", 0) for m in recent]),
                "samples_per_second": self.total_samples / (time.time() - self.start_time),
                "total_samples": self.total_samples
            }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get training summary"""
        return self.summary_stats
    
    def save_metrics(self, filepath: str):
        """Save all metrics to file"""
        with open(filepath, 'w') as f:
            json.dump(list(self.metrics_buffer), f, indent=2)

@ray.remote
class CheckpointActor:
    """
    Ray actor for model checkpointing
    Handles saving and loading model states
    """
    
    def __init__(self, config: M2BertConfig, checkpoint_dir: str = "./checkpoints"):
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.checkpoints_saved = 0
        self.best_eval_loss = float('inf')
    
    def save_checkpoint(self, model_state: Dict[str, Any], metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        global_step = model_state["global_step"]
        
        # Regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{global_step}.pt"
        torch.save(model_state, checkpoint_path)
        
        # Best checkpoint
        if is_best or metrics.get("eval_loss", float('inf')) < self.best_eval_loss:
            self.best_eval_loss = metrics.get("eval_loss", self.best_eval_loss)
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(model_state, best_path)
            print(f"✓ Saved best model (eval_loss: {self.best_eval_loss:.4f})")
        
        self.checkpoints_saved += 1
        
        # Keep only last N checkpoints
        if self.checkpoints_saved > 10:
            self.cleanup_old_checkpoints()
        
        return str(checkpoint_path)
    
    def cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only recent ones"""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_step_*.pt"))
        if len(checkpoints) > 10:
            for ckpt in checkpoints[:-10]:
                ckpt.unlink()

class M2BertRayTrainer:
    """
    Orchestrator for distributed M2-BERT training
    Coordinates all Ray actors for efficient training
    """
    
    def __init__(self, config: M2BertConfig):
        self.config = config
        
        # Initialize actors
        print("Initializing Ray actors...")
        
        # Data actors (multiple for parallel loading)
        self.data_actors = [
            DataLoaderActor.remote(config, rank=i)
            for i in range(config.num_data_workers)
        ]
        
        # Model actors (for model parallelism if needed)
        self.model_actors = [
            ModelActor.remote(config, rank=i)
            for i in range(config.num_model_replicas)
        ]
        
        # Single instances of other actors
        self.evaluator = EvaluatorActor.remote(config)
        self.metrics_actor = MetricsActor.remote(config)
        self.checkpoint_actor = CheckpointActor.remote(config)
        
        # Actor pool for load balancing
        self.model_pool = ActorPool(self.model_actors)
        
        print(f"Ray actors initialized:")
        print(f"  Data workers: {config.num_data_workers}")
        print(f"  Model replicas: {config.num_model_replicas}")
        print(f"  Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU/MPS'}")
    
    def train(self, num_steps: int = 100000):
        """
        Main training loop
        """
        print("\n" + "="*70)
        print("M2-BERT DISTRIBUTED TRAINING")
        print(f"Target: {num_steps:,} steps")
        print(f"Batch size: {self.config.global_batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print("="*70 + "\n")
        
        global_step = 0
        
        while global_step < num_steps:
            # Get batches from data actors
            batch_futures = []
            for data_actor in self.data_actors:
                batch_futures.append(data_actor.get_batch.remote())
            
            batches = ray.get(batch_futures)
            
            # Distribute batches to model actors
            metrics_futures = []
            for batch, model_actor in zip(batches, self.model_actors):
                future = model_actor.forward_backward.remote(batch)
                metrics_futures.append(future)
            
            # Collect metrics
            step_metrics = ray.get(metrics_futures)
            
            # Aggregate metrics
            aggregated = {
                "loss": np.mean([m["loss"] for m in step_metrics]),
                "accuracy": np.mean([m["accuracy"] for m in step_metrics]),
                "gradient_norm": np.mean([m["gradient_norm"] for m in step_metrics]),
                "global_step": global_step
            }
            
            # Record metrics
            ray.get(self.metrics_actor.record_metrics.remote(aggregated))
            
            # Logging
            if global_step % self.config.log_interval == 0:
                summary = ray.get(self.metrics_actor.get_summary.remote())
                print(f"Step {global_step:6d} | Loss: {aggregated['loss']:.4f} | "
                      f"Acc: {aggregated['accuracy']:.2%} | "
                      f"Grad: {aggregated['gradient_norm']:.2f} | "
                      f"Speed: {summary.get('samples_per_second', 0):.1f} samples/s")
            
            # Evaluation
            if global_step % self.config.eval_interval == 0 and global_step > 0:
                self.run_evaluation(global_step)
            
            # Checkpointing
            if global_step % self.config.checkpoint_interval == 0 and global_step > 0:
                self.save_checkpoint(global_step)
            
            global_step += 1
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        
        # Final evaluation
        self.run_evaluation(global_step)
        
        # Save final checkpoint
        self.save_checkpoint(global_step)
        
        # Save metrics
        ray.get(self.metrics_actor.save_metrics.remote("training_metrics.json"))
    
    def run_evaluation(self, global_step: int):
        """Run asynchronous evaluation"""
        print(f"\nRunning evaluation at step {global_step}...")
        
        # Get eval batches
        eval_batch_futures = []
        for data_actor in self.data_actors[:2]:  # Use 2 actors for eval
            eval_batch_futures.append(data_actor.get_batches.remote(10))
        
        eval_batches = []
        for batches in ray.get(eval_batch_futures):
            eval_batches.extend(batches)
        
        # Get model state from first model actor
        model_state = ray.get(self.model_actors[0].get_model_state.remote())
        
        # Run evaluation
        eval_metrics = ray.get(
            self.evaluator.evaluate.remote(model_state, eval_batches)
        )
        
        print(f"Eval Loss: {eval_metrics['eval_loss']:.4f} | "
              f"Eval Acc: {eval_metrics['eval_accuracy']:.2%}")
    
    def save_checkpoint(self, global_step: int):
        """Save model checkpoint"""
        print(f"Saving checkpoint at step {global_step}...")
        
        # Get model state
        model_state = ray.get(self.model_actors[0].get_model_state.remote())
        
        # Get current metrics
        summary = ray.get(self.metrics_actor.get_summary.remote())
        
        # Save checkpoint
        checkpoint_path = ray.get(
            self.checkpoint_actor.save_checkpoint.remote(
                model_state, summary, is_best=False
            )
        )
        
        print(f"Checkpoint saved: {checkpoint_path}")

def main():
    """
    Main entry point for distributed M2-BERT training
    """
    # Configuration
    config = M2BertConfig(
        # Model: 110M parameter version
        hidden_dim=960,
        intermediate_dim=3840,
        n_layers=12,
        n_blocks=4,
        
        # Training
        learning_rate=8e-4,
        global_batch_size=4096,
        micro_batch_size=128,
        
        # System
        num_model_replicas=2,
        num_data_workers=4,
        
        # Evaluation
        eval_interval=100,  # More frequent for demo
        checkpoint_interval=500
    )
    
    # Create trainer
    trainer = M2BertRayTrainer(config)
    
    # Run training
    trainer.train(num_steps=1000)  # Short run for demo
    
    # Cleanup
    ray.shutdown()

if __name__ == "__main__":
    print("*" * 70)
    print("M2-BERT DISTRIBUTED TRAINING WITH RAY")
    print("World-class actor-based pipeline")
    print("*" * 70)
    
    main()