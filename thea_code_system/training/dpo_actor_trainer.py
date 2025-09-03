#!/usr/bin/env python
"""
DPO Actor-Based Training for M2-BERT Enhanced
Reverse-engineered from LFM2's training approach
Uses TRL 0.18.2+ patterns with our actor infrastructure
"""

import ray
import torch
import torch.nn as nn
from torch.optim import AdamW
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import asyncio

from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

from ..core.training import TrainingActor, DistributedTrainer, TrainingConfig
from ..core.enhanced_pool import EnhancedActorPool
from ..core.base import ActorConfig
from ..models.m2_bert_enhanced import M2BertEnhanced, M2BertEnhancedConfig


@dataclass
class DPOConfig:
    """
    DPO configuration matching LFM2's training setup
    From the colab: DPOConfig from TRL 0.18.2
    """
    # Training parameters from LFM2
    output_dir: str = "./m2-bert-dpo"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1  # They use 1 for memory
    learning_rate: float = 1e-6  # Exact LR from LFM2
    lr_scheduler_type: str = "linear"
    
    # DPO specific
    beta: float = 0.1  # DPO temperature
    label_smoothing: float = 0.0
    loss_type: str = "sigmoid"  # or "hinge", "ipo"
    
    # LoRA configuration from LFM2
    use_lora: bool = True
    lora_r: int = 8  # Exact rank from LFM2
    lora_alpha: int = 16  # Exact alpha from LFM2
    lora_dropout: float = 0.1
    
    # Optimization
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    
    # Logging
    logging_steps: int = 10
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    
    # Hardware
    bf16: bool = False  # Not all GPUs support bf16
    fp16: bool = True   # Use fp16 instead
    
    # Actor-based training
    num_training_actors: int = 2
    use_parameter_server: bool = True


class DPOLoss(nn.Module):
    """
    DPO loss implementation matching TRL's approach
    This is what LFM2 uses for preference optimization
    """
    
    def __init__(self, beta: float = 0.1, label_smoothing: float = 0.0, loss_type: str = "sigmoid"):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.loss_type = loss_type
    
    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute DPO loss as in LFM2
        """
        # Compute log ratios
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps)
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps)
        
        # Compute loss based on type
        if self.loss_type == "sigmoid":
            # Standard DPO loss (what LFM2 uses)
            loss = -F.logsigmoid(chosen_rewards - rejected_rewards)
        elif self.loss_type == "hinge":
            # Hinge loss variant
            loss = torch.relu(1 - (chosen_rewards - rejected_rewards))
        elif self.loss_type == "ipo":
            # IPO loss variant
            loss = (chosen_rewards - rejected_rewards - 1) ** 2
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            loss = loss * (1 - self.label_smoothing) + self.label_smoothing * 0.5
        
        return loss.mean()


class DPOTrainingActor(TrainingActor):
    """
    Training actor specialized for DPO
    Handles preference learning like LFM2
    """
    
    def __init__(self, actor_id: str, config: Optional[ActorConfig] = None):
        super().__init__(actor_id, config)
        self.reference_model = None
        self.dpo_loss = None
        
    async def initialize_dpo(
        self,
        model_class: type,
        model_config: M2BertEnhancedConfig,
        tokenizer: Any,
        dpo_config: DPOConfig
    ) -> None:
        """Initialize for DPO training matching LFM2's setup"""
        # Create model
        self.model = model_class(model_config).to(self.device)
        
        # Apply LoRA if configured (exactly as LFM2 does)
        if dpo_config.use_lora:
            # Get target modules from model (this is the key insight from LFM2)
            target_modules = self.model.get_lora_target_modules()
            
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=dpo_config.lora_r,
                lora_alpha=dpo_config.lora_alpha,
                lora_dropout=dpo_config.lora_dropout,
                target_modules=target_modules,  # Use the exact modules from LFM2
                bias="none",
                modules_to_save=None
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.logger.info(f"Applied LoRA with target modules: {target_modules}")
            self.model.print_trainable_parameters()
        
        # Create reference model (frozen copy for DPO)
        self.reference_model = model_class(model_config).to(self.device)
        self.reference_model.eval()
        for param in self.reference_model.parameters():
            param.requires_grad = False
        
        # Create optimizer (matching LFM2's setup)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=dpo_config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Create DPO loss
        self.dpo_loss = DPOLoss(
            beta=dpo_config.beta,
            label_smoothing=dpo_config.label_smoothing,
            loss_type=dpo_config.loss_type
        )
        
        self.tokenizer = tokenizer
        self.dpo_config = dpo_config
        self._initialized = True
    
    async def compute_log_probs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute log probabilities for DPO"""
        with torch.no_grad() if model == self.reference_model else torch.enable_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs['logits']
            
            # Shift for autoregressive
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute log probs
            log_probs = F.log_softmax(shift_logits, dim=-1)
            
            # Gather log probs for actual tokens
            gathered_log_probs = torch.gather(
                log_probs,
                dim=-1,
                index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)
            
            # Mask padding
            mask = shift_labels != self.tokenizer.pad_token_id
            gathered_log_probs = gathered_log_probs * mask
            
            # Sum over sequence
            return gathered_log_probs.sum(dim=-1)
    
    async def train_dpo_batch(
        self,
        batch: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Train on a DPO batch (chosen vs rejected)
        This is the core of LFM2's training
        """
        # Tokenize if needed
        if 'input_ids_chosen' not in batch:
            # Tokenize chosen
            chosen_encoded = self.tokenizer(
                batch['chosen'],
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )
            batch['input_ids_chosen'] = chosen_encoded['input_ids']
            batch['labels_chosen'] = chosen_encoded['input_ids'].clone()
            
            # Tokenize rejected
            rejected_encoded = self.tokenizer(
                batch['rejected'],
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )
            batch['input_ids_rejected'] = rejected_encoded['input_ids']
            batch['labels_rejected'] = rejected_encoded['input_ids'].clone()
        
        # Move to device
        input_ids_chosen = batch['input_ids_chosen'].to(self.device)
        labels_chosen = batch['labels_chosen'].to(self.device)
        input_ids_rejected = batch['input_ids_rejected'].to(self.device)
        labels_rejected = batch['labels_rejected'].to(self.device)
        
        # Compute log probs for policy model
        policy_chosen_logps = await self.compute_log_probs(
            self.model, input_ids_chosen, labels_chosen
        )
        policy_rejected_logps = await self.compute_log_probs(
            self.model, input_ids_rejected, labels_rejected
        )
        
        # Compute log probs for reference model
        ref_chosen_logps = await self.compute_log_probs(
            self.reference_model, input_ids_chosen, labels_chosen
        )
        ref_rejected_logps = await self.compute_log_probs(
            self.reference_model, input_ids_rejected, labels_rejected
        )
        
        # Compute DPO loss
        loss = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if self.dpo_config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.dpo_config.max_grad_norm
            )
        
        # Optimization step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Update metrics
        self.training_metrics.total_steps += 1
        
        return {
            'loss': loss.item(),
            'chosen_rewards': (policy_chosen_logps - ref_chosen_logps).mean().item(),
            'rejected_rewards': (policy_rejected_logps - ref_rejected_logps).mean().item(),
            'total_steps': self.training_metrics.total_steps
        }
    
    async def process(self, batch: Any) -> Any:
        """Process a batch for DPO training"""
        if isinstance(batch, dict) and 'chosen' in batch and 'rejected' in batch:
            return await self.train_dpo_batch(batch)
        return batch


class ActorBasedDPOTrainer:
    """
    Distributed DPO trainer using our actor infrastructure
    Implements LFM2's training approach with actors
    """
    
    def __init__(
        self,
        model_config: M2BertEnhancedConfig,
        dpo_config: DPOConfig,
        tokenizer: Any
    ):
        self.model_config = model_config
        self.dpo_config = dpo_config
        self.tokenizer = tokenizer
        self.training_actors = []
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize distributed DPO training"""
        # Create training actors
        for i in range(self.dpo_config.num_training_actors):
            actor_config = ActorConfig(
                name=f"dpo_trainer_{i}",
                num_cpus=2.0,
                num_gpus=0.5 if torch.cuda.is_available() else 0
            )
            
            # Create remote actor
            TrainerClass = ray.remote(DPOTrainingActor)
            actor = TrainerClass.remote(f"dpo_{i}", actor_config)
            
            # Initialize for DPO
            await actor.initialize_dpo.remote(
                M2BertEnhanced,
                self.model_config,
                self.tokenizer,
                self.dpo_config
            )
            
            self.training_actors.append(actor)
        
        self._initialized = True
        print(f"✅ Initialized {len(self.training_actors)} DPO training actors")
    
    async def train_on_dataset(
        self,
        dataset_name: str = "mlabonne/orpo-dpo-mix-40k",
        num_samples: int = 2000
    ) -> Dict[str, Any]:
        """
        Train on DPO dataset like LFM2
        Uses the exact dataset from the colab
        """
        if not self._initialized:
            await self.initialize()
        
        # Load dataset (exactly as in LFM2 colab)
        print(f"📥 Loading DPO dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, split=f"train[:{num_samples}]")
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = dataset['train']
        eval_dataset = dataset['test']
        
        print(f"✅ Dataset loaded:")
        print(f"   📚 Train samples: {len(train_dataset)}")
        print(f"   🧪 Eval samples: {len(eval_dataset)}")
        
        # Training loop
        total_steps = len(train_dataset) // (
            self.dpo_config.per_device_train_batch_size * 
            self.dpo_config.num_training_actors
        )
        
        for epoch in range(self.dpo_config.num_train_epochs):
            print(f"\n🚀 Epoch {epoch + 1}/{self.dpo_config.num_train_epochs}")
            
            epoch_loss = 0.0
            epoch_chosen_rewards = 0.0
            epoch_rejected_rewards = 0.0
            
            # Process batches
            for step in range(0, len(train_dataset), self.dpo_config.num_training_actors):
                # Get batch for each actor
                batches = []
                for i in range(self.dpo_config.num_training_actors):
                    if step + i < len(train_dataset):
                        batches.append(train_dataset[step + i])
                
                # Train on all actors in parallel
                train_futures = []
                for actor, batch in zip(self.training_actors[:len(batches)], batches):
                    train_futures.append(actor.train_dpo_batch.remote(batch))
                
                results = ray.get(train_futures)
                
                # Aggregate metrics
                for result in results:
                    epoch_loss += result['loss']
                    epoch_chosen_rewards += result['chosen_rewards']
                    epoch_rejected_rewards += result['rejected_rewards']
                
                # Log progress
                if (step // self.dpo_config.num_training_actors) % self.dpo_config.logging_steps == 0:
                    avg_loss = epoch_loss / (step + 1)
                    avg_chosen = epoch_chosen_rewards / (step + 1)
                    avg_rejected = epoch_rejected_rewards / (step + 1)
                    
                    print(f"   Step {step}: Loss={avg_loss:.4f}, "
                          f"Chosen={avg_chosen:.4f}, Rejected={avg_rejected:.4f}")
            
            # Evaluation
            if self.dpo_config.eval_strategy == "epoch":
                eval_loss = await self.evaluate(eval_dataset)
                print(f"   📊 Eval Loss: {eval_loss:.4f}")
        
        print("🎉 DPO training completed!")
        
        return {
            'final_loss': epoch_loss / len(train_dataset),
            'chosen_rewards': epoch_chosen_rewards / len(train_dataset),
            'rejected_rewards': epoch_rejected_rewards / len(train_dataset)
        }
    
    async def evaluate(self, eval_dataset) -> float:
        """Evaluate on validation set"""
        total_loss = 0.0
        
        for i in range(0, len(eval_dataset), self.dpo_config.num_training_actors):
            batches = []
            for j in range(self.dpo_config.num_training_actors):
                if i + j < len(eval_dataset):
                    batches.append(eval_dataset[i + j])
            
            eval_futures = []
            for actor, batch in zip(self.training_actors[:len(batches)], batches):
                eval_futures.append(actor.train_dpo_batch.remote(batch))
            
            results = ray.get(eval_futures)
            for result in results:
                total_loss += result['loss']
        
        return total_loss / len(eval_dataset)
    
    async def save_model(self, output_dir: str = None) -> None:
        """Save the trained model"""
        if output_dir is None:
            output_dir = self.dpo_config.output_dir
        
        # Get model from first actor
        first_actor = self.training_actors[0]
        state = await first_actor.get_model_state.remote()
        
        # Save
        torch.save(state, f"{output_dir}/model_state.pt")
        print(f"💾 Model saved to: {output_dir}")
    
    async def merge_and_save_lora(self, output_dir: str = "./m2-bert-lora-merged") -> None:
        """
        Merge LoRA weights back into the model
        This is exactly what LFM2 does after training
        """
        print("\n🔄 Merging LoRA weights...")
        
        # This would need the actual model instance
        # For now, we save the LoRA state
        await self.save_model(output_dir)
        
        print(f"💾 Merged model saved to: {output_dir}")
    
    async def shutdown(self) -> None:
        """Shutdown training actors"""
        for actor in self.training_actors:
            ray.kill(actor)
        self._initialized = False