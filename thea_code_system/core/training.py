#!/usr/bin/env python
"""
Training Infrastructure for Actor-Based Learning
Carefully manages async flows and gradient synchronization
"""

import ray
import torch
import torch.nn as nn
from torch.optim import Optimizer
import asyncio
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import time
import numpy as np

from .enhanced_pool import EnhancedActor
from .base import ActorConfig
from .scalars import ScalarOperations


@dataclass
class TrainingConfig:
    """Configuration for distributed training"""
    batch_size: int = 32
    learning_rate: float = 1e-4
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    checkpoint_every: int = 1000
    log_every: int = 100
    use_mixed_precision: bool = False
    device: str = "auto"  # auto, cpu, cuda, mps


@dataclass 
class TrainingMetrics:
    """Metrics tracked during training"""
    total_steps: int = 0
    total_loss: float = 0.0
    gradient_norm: float = 0.0
    learning_rate: float = 0.0
    throughput: float = 0.0  # samples/sec
    last_checkpoint: int = 0
    losses: List[float] = field(default_factory=list)
    
    def update(self, loss: float, grad_norm: float, lr: float, batch_size: int, time_taken: float):
        """Update metrics"""
        self.total_steps += 1
        self.total_loss += loss
        self.gradient_norm = grad_norm
        self.learning_rate = lr
        self.throughput = batch_size / time_taken if time_taken > 0 else 0
        self.losses.append(loss)
        
    @property
    def average_loss(self) -> float:
        """Get average loss"""
        if not self.losses:
            return 0.0
        return sum(self.losses[-100:]) / len(self.losses[-100:])  # Last 100 steps


class TrainingActor(EnhancedActor):
    """
    Actor specialized for training neural networks.
    Handles model state, gradients, and optimization.
    """
    
    def __init__(self, actor_id: str, config: Optional[ActorConfig] = None):
        super().__init__(actor_id, config)
        self.model = None
        self.optimizer = None
        self.training_config = None
        self.training_metrics = TrainingMetrics()
        self.gradient_buffer = []
        
    async def initialize_training(
        self,
        model_class: type,
        model_kwargs: Dict,
        optimizer_class: type,
        optimizer_kwargs: Dict,
        training_config: TrainingConfig
    ) -> None:
        """Initialize model and optimizer for training"""
        await self.initialize()  # Base initialization
        
        self.training_config = training_config
        
        # Create model on device
        self.model = model_class(**model_kwargs).to(self.device)
        
        # Create optimizer
        self.optimizer = optimizer_class(
            self.model.parameters(),
            **optimizer_kwargs
        )
        
        # Setup mixed precision if requested
        if training_config.use_mixed_precision and self.device.type in ['cuda', 'mps']:
            self.scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None
        else:
            self.scaler = None
            
        self.logger.info(f"Training actor {self.actor_id} initialized on {self.device}")
        
    async def forward_pass(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Perform forward pass on batch.
        Returns loss and additional outputs.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call initialize_training first.")
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        # Forward pass (with or without mixed precision)
        if self.scaler:
            with torch.cuda.amp.autocast():
                outputs = self.model(**batch)
        else:
            outputs = self.model(**batch)
        
        # Extract loss (assuming model returns dict with 'loss' key)
        if isinstance(outputs, dict):
            loss = outputs.get('loss')
        elif isinstance(outputs, torch.Tensor):
            loss = outputs
        else:
            loss = outputs[0] if isinstance(outputs, tuple) else outputs.loss
            
        return loss, outputs
    
    async def backward_pass(self, loss: torch.Tensor) -> float:
        """
        Perform backward pass and return gradient norm.
        Handles gradient accumulation.
        """
        # Scale loss for gradient accumulation
        if self.training_config.gradient_accumulation_steps > 1:
            loss = loss / self.training_config.gradient_accumulation_steps
        
        # Backward pass
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Calculate gradient norm
        grad_norm = self._calculate_gradient_norm()
        
        return grad_norm
    
    async def optimization_step(self) -> bool:
        """
        Perform optimization step if gradients are accumulated.
        Returns True if step was taken.
        """
        self.training_metrics.total_steps += 1
        
        # Check if we should take a step
        if self.training_metrics.total_steps % self.training_config.gradient_accumulation_steps != 0:
            return False
        
        # Clip gradients
        if self.training_config.max_grad_norm > 0:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.training_config.max_grad_norm
            )
        
        # Optimization step
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        # Clear gradients
        self.optimizer.zero_grad()
        
        return True
    
    async def process(self, batch: Any) -> Any:
        """
        Required abstract method - delegates to train_batch.
        """
        if isinstance(batch, dict) and ('inputs' in batch or 'input_ids' in batch):
            return await self.train_batch(batch)
        return batch
    
    async def train_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Complete training step for a batch.
        This is the main entry point for training.
        """
        start_time = time.perf_counter()
        
        # Forward pass
        loss, outputs = await self.forward_pass(batch)
        
        # Backward pass
        grad_norm = await self.backward_pass(loss)
        
        # Optimization step (may not happen every batch due to accumulation)
        step_taken = await self.optimization_step()
        
        # Update metrics
        elapsed = time.perf_counter() - start_time
        batch_size = next(iter(batch.values())).size(0)
        
        self.training_metrics.update(
            loss=loss.item(),
            grad_norm=grad_norm,
            lr=self.optimizer.param_groups[0]['lr'],
            batch_size=batch_size,
            time_taken=elapsed
        )
        
        return {
            'loss': loss.item(),
            'grad_norm': grad_norm,
            'step_taken': step_taken,
            'throughput': self.training_metrics.throughput,
            'total_steps': self.training_metrics.total_steps
        }
    
    def _calculate_gradient_norm(self) -> float:
        """Calculate L2 norm of gradients"""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    async def get_model_state(self) -> Dict[str, Any]:
        """Get model state for checkpointing or sharing"""
        if self.model is None:
            return {}
        
        return {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_metrics': self.training_metrics,
            'training_config': self.training_config
        }
    
    async def load_model_state(self, state: Dict[str, Any]) -> None:
        """Load model state from checkpoint"""
        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.training_metrics = state.get('training_metrics', TrainingMetrics())
        
    async def sync_gradients(self, gradients: List[torch.Tensor]) -> None:
        """Synchronize gradients from other actors"""
        if self.model is None:
            return
        
        # Average gradients from all actors
        for param, grad in zip(self.model.parameters(), gradients):
            if param.grad is None:
                param.grad = grad.clone()
            else:
                param.grad += grad
                
    async def get_gradients(self) -> List[torch.Tensor]:
        """Get gradients for synchronization"""
        if self.model is None:
            return []
        
        return [p.grad.clone() if p.grad is not None else torch.zeros_like(p)
                for p in self.model.parameters()]


@ray.remote
class ParameterServerActor:
    """
    Parameter server for weight synchronization across training actors.
    Manages the authoritative copy of model weights.
    """
    
    def __init__(self):
        self.model_state = None
        self.optimizer_state = None
        self.global_step = 0
        self.accumulated_gradients = []
        self.num_actors = 0
        self.device = self._setup_device()
        
    def _setup_device(self) -> torch.device:
        """Setup compute device"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def initialize(self, model_state: Dict, num_actors: int) -> None:
        """Initialize parameter server with model state"""
        self.model_state = model_state
        self.num_actors = num_actors
        self.accumulated_gradients = []
        
    def get_weights(self) -> Dict:
        """Get current model weights"""
        return self.model_state
    
    def accumulate_gradients(self, gradients: List[torch.Tensor], actor_id: str) -> None:
        """Accumulate gradients from an actor"""
        # Move gradients to parameter server device
        gradients = [g.to(self.device) for g in gradients]
        
        if not self.accumulated_gradients:
            self.accumulated_gradients = gradients
        else:
            for i, grad in enumerate(gradients):
                self.accumulated_gradients[i] += grad
    
    def average_gradients(self) -> List[torch.Tensor]:
        """Average accumulated gradients"""
        if not self.accumulated_gradients:
            return []
        
        averaged = []
        for grad in self.accumulated_gradients:
            averaged.append(grad / self.num_actors)
        
        # Clear accumulator
        self.accumulated_gradients = []
        
        return averaged
    
    def update_weights(self, new_state: Dict) -> None:
        """Update model weights after optimization"""
        self.model_state = new_state
        self.global_step += 1
        
    def get_global_step(self) -> int:
        """Get current global training step"""
        return self.global_step


class DataLoaderActor(EnhancedActor):
    """
    Actor responsible for data loading and batch preparation.
    Ensures data is ready when training actors need it.
    """
    
    def __init__(self, actor_id: str, config: Optional[ActorConfig] = None):
        super().__init__(actor_id, config)
        self.data_queue = asyncio.Queue(maxsize=100)
        self.dataset = None
        self.batch_size = 32
        self.current_epoch = 0
        
    async def initialize_data(
        self,
        dataset: Any,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0
    ) -> None:
        """Initialize data loading"""
        await self.initialize()
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Create data loader
        self.data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda"
        )
        
        self.data_iter = iter(self.data_loader)
        
    async def get_batch(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get next batch of data"""
        try:
            batch = next(self.data_iter)
        except StopIteration:
            # Reset iterator for next epoch
            self.current_epoch += 1
            self.data_iter = iter(self.data_loader)
            batch = next(self.data_iter)
        
        # Convert to dict format if needed
        if isinstance(batch, (list, tuple)):
            batch = {'inputs': batch[0], 'labels': batch[1]}
        
        return batch
    
    async def prefetch_batches(self, num_batches: int = 10) -> List[Dict]:
        """Prefetch multiple batches for efficiency"""
        batches = []
        for _ in range(num_batches):
            batch = await self.get_batch()
            if batch:
                batches.append(batch)
        return batches
    
    async def process(self, request: str) -> Any:
        """Process data requests"""
        if request == "get_batch":
            return await self.get_batch()
        elif request == "prefetch":
            return await self.prefetch_batches()
        else:
            return None


class DistributedTrainer:
    """
    Orchestrates distributed training across multiple actors.
    Manages training flow, synchronization, and checkpointing.
    """
    
    def __init__(
        self,
        num_training_actors: int = 2,
        training_config: Optional[TrainingConfig] = None
    ):
        self.num_training_actors = num_training_actors
        self.training_config = training_config or TrainingConfig()
        self.training_actors = []
        self.parameter_server = None
        self.data_loader_actor = None
        self._initialized = False
        
    async def initialize(
        self,
        model_class: type,
        model_kwargs: Dict,
        optimizer_class: type,
        optimizer_kwargs: Dict,
        dataset: Any
    ) -> None:
        """Initialize distributed training system"""
        
        # Create parameter server
        self.parameter_server = ParameterServerActor.remote()
        
        # Create training actors
        for i in range(self.num_training_actors):
            config = ActorConfig(
                name=f"trainer_{i}",
                num_cpus=1.0,
                num_gpus=0.25 if torch.cuda.is_available() else 0
            )
            
            TrainerClass = ray.remote(TrainingActor)
            actor = TrainerClass.remote(f"trainer_{i}", config)
            
            # Initialize training on actor
            await actor.initialize_training.remote(
                model_class,
                model_kwargs,
                optimizer_class,
                optimizer_kwargs,
                self.training_config
            )
            
            self.training_actors.append(actor)
        
        # Initialize parameter server with first actor's state
        first_state = await self.training_actors[0].get_model_state.remote()
        ray.get(
            self.parameter_server.initialize.remote(
                first_state['model_state_dict'],
                self.num_training_actors
            )
        )
        
        # Create data loader actor
        DataLoaderClass = ray.remote(DataLoaderActor)
        self.data_loader_actor = DataLoaderClass.remote("data_loader", None)
        await self.data_loader_actor.initialize_data.remote(
            dataset,
            self.training_config.batch_size,
            shuffle=True
        )
        
        self._initialized = True
        print(f"✅ Distributed trainer initialized with {self.num_training_actors} actors")
    
    async def train_epoch(self, num_steps: int) -> Dict[str, Any]:
        """
        Train for specified number of steps.
        Coordinates actors and handles synchronization.
        """
        if not self._initialized:
            raise RuntimeError("Trainer not initialized")
        
        epoch_metrics = {
            'total_loss': 0.0,
            'steps': 0,
            'throughput': 0.0
        }
        
        for step in range(num_steps):
            start_time = time.perf_counter()
            
            # Get batches for all actors
            batch_futures = []
            for _ in range(self.num_training_actors):
                batch_futures.append(
                    self.data_loader_actor.get_batch.remote()
                )
            batches = ray.get(batch_futures)
            
            # Train on all actors in parallel
            train_futures = []
            for actor, batch in zip(self.training_actors, batches):
                train_futures.append(
                    actor.train_batch.remote(batch)
                )
            
            results = ray.get(train_futures)
            
            # Synchronize gradients if needed
            if step % self.training_config.gradient_accumulation_steps == 0:
                await self._synchronize_gradients()
            
            # Update metrics
            avg_loss = sum(r['loss'] for r in results) / len(results)
            epoch_metrics['total_loss'] += avg_loss
            epoch_metrics['steps'] += 1
            
            elapsed = time.perf_counter() - start_time
            epoch_metrics['throughput'] = self.training_config.batch_size * self.num_training_actors / elapsed
            
            # Log progress
            if step % self.training_config.log_every == 0:
                print(f"Step {step}: Loss={avg_loss:.4f}, Throughput={epoch_metrics['throughput']:.1f} samples/sec")
            
            # Checkpoint if needed
            if step % self.training_config.checkpoint_every == 0 and step > 0:
                await self.checkpoint(f"checkpoint_step_{step}")
        
        return epoch_metrics
    
    async def _synchronize_gradients(self) -> None:
        """Synchronize gradients across all training actors"""
        # Collect gradients from all actors
        grad_futures = []
        for actor in self.training_actors:
            grad_futures.append(actor.get_gradients.remote())
        
        all_gradients = ray.get(grad_futures)
        
        # Average gradients on parameter server
        for grads, actor_id in zip(all_gradients, range(len(self.training_actors))):
            ray.get(
                self.parameter_server.accumulate_gradients.remote(grads, f"actor_{actor_id}")
            )
        
        averaged_grads = ray.get(self.parameter_server.average_gradients.remote())
        
        # Distribute averaged gradients back to actors
        sync_futures = []
        for actor in self.training_actors:
            sync_futures.append(actor.sync_gradients.remote(averaged_grads))
        
        ray.get(sync_futures)
    
    async def checkpoint(self, checkpoint_name: str) -> None:
        """Save training checkpoint"""
        # Get state from first actor
        state = await self.training_actors[0].get_model_state.remote()
        
        # Save to disk (simplified - in production use proper storage)
        torch.save(state, f"{checkpoint_name}.pt")
        print(f"✅ Checkpoint saved: {checkpoint_name}")
    
    async def shutdown(self) -> None:
        """Shutdown training system"""
        for actor in self.training_actors:
            ray.kill(actor)
        
        if self.parameter_server:
            ray.kill(self.parameter_server)
        
        if self.data_loader_actor:
            ray.kill(self.data_loader_actor)
        
        self._initialized = False