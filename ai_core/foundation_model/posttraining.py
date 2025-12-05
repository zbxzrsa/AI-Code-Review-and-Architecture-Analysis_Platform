"""
Post-training Pipeline for Foundation Model

Implements:
1. Supervised Fine-Tuning (SFT) - High-quality dialogue data
2. RLHF (Reinforcement Learning from Human Feedback)
   - Reward Model training
   - PPO optimization
3. Constitutional AI - Safety and value alignment
4. Direct Preference Optimization (DPO) - Alternative to RLHF

Target: Align model with human preferences and safety requirements
"""

import copy
import logging
import math
import random
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class AlignmentMethod(str, Enum):
    """Alignment training methods."""
    SFT = "sft"  # Supervised Fine-Tuning
    RLHF = "rlhf"  # Reinforcement Learning from Human Feedback
    DPO = "dpo"  # Direct Preference Optimization
    CONSTITUTIONAL = "constitutional"  # Constitutional AI


@dataclass
class PosttrainingConfig:
    """Post-training configuration."""
    # General
    method: AlignmentMethod = AlignmentMethod.RLHF
    
    # SFT settings
    sft_learning_rate: float = 2e-5
    sft_epochs: int = 3
    sft_batch_size: int = 8
    sft_max_length: int = 4096
    
    # RLHF settings
    reward_learning_rate: float = 1e-5
    reward_epochs: int = 1
    ppo_learning_rate: float = 1e-5
    ppo_epochs: int = 4
    ppo_batch_size: int = 32
    ppo_clip_ratio: float = 0.2
    ppo_value_clip: float = 0.2
    ppo_entropy_coef: float = 0.01
    ppo_vf_coef: float = 0.5
    kl_target: float = 0.02
    kl_coef: float = 0.1
    gamma: float = 1.0
    gae_lambda: float = 0.95
    
    # DPO settings
    dpo_learning_rate: float = 1e-6
    dpo_beta: float = 0.1
    dpo_epochs: int = 3
    
    # Constitutional AI
    constitution_path: Optional[str] = None
    critique_model: Optional[str] = None
    revision_model: Optional[str] = None
    
    # General training
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # Response generation
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass
class Conversation:
    """Single conversation for training."""
    conversation_id: str
    messages: List[Dict[str, str]]  # [{"role": "user/assistant", "content": "..."}]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreferencePair:
    """Preference pair for RLHF/DPO training."""
    prompt: str
    chosen: str  # Preferred response
    rejected: str  # Non-preferred response
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Supervised Fine-Tuning (SFT)
# =============================================================================

class SFTDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    
    def __init__(
        self,
        conversations: List[Conversation],
        tokenizer: Any,
        max_length: int = 4096,
    ):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.conversations)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        conv = self.conversations[idx]
        
        # Format conversation
        text = self._format_conversation(conv.messages)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze(),
        }
    
    def _format_conversation(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into training text."""
        formatted = []
        
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            if role == 'user':
                formatted.append(f"User: {content}")
            elif role == 'assistant':
                formatted.append(f"Assistant: {content}")
            elif role == 'system':
                formatted.append(f"System: {content}")
        
        return "\n\n".join(formatted)


class SFTTrainer:
    """
    Supervised Fine-Tuning Trainer
    
    Trains the model on high-quality dialogue data
    to learn instruction-following behavior.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: PosttrainingConfig,
        tokenizer: Any,
    ):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.sft_learning_rate,
            weight_decay=config.weight_decay,
        )
        
        self.device = next(model.parameters()).device
    
    def train(
        self,
        train_data: List[Conversation],
        eval_data: Optional[List[Conversation]] = None,
    ) -> Dict[str, float]:
        """
        Train the model with SFT.
        
        Args:
            train_data: Training conversations
            eval_data: Optional evaluation conversations
        """
        # Create dataset and dataloader
        dataset = SFTDataset(
            train_data,
            self.tokenizer,
            self.config.sft_max_length,
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.sft_batch_size,
            shuffle=True,
            drop_last=True,
        )
        
        # Training loop
        self.model.train()
        total_loss = 0.0
        total_steps = 0
        
        for epoch in range(self.config.sft_epochs):
            epoch_loss = 0.0
            
            for batch_idx, batch in enumerate(dataloader):
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                
                logits = outputs['logits']
                
                # Compute loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
                
                # Backward pass
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                epoch_loss += loss.item() * self.config.gradient_accumulation_steps
                total_steps += 1
            
            avg_loss = epoch_loss / len(dataloader)
            total_loss += avg_loss
            
            logger.info(f"SFT Epoch {epoch + 1}/{self.config.sft_epochs}, Loss: {avg_loss:.4f}")
        
        return {
            'sft_loss': total_loss / self.config.sft_epochs,
            'total_steps': total_steps,
        }


# =============================================================================
# Reward Model
# =============================================================================

class RewardModel(nn.Module):
    """
    Reward Model for RLHF
    
    Trained to predict human preferences between responses.
    Output: scalar reward score for a (prompt, response) pair.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        hidden_size: int = 4096,
    ):
        super().__init__()
        
        # Use base model for encoding
        self.base_model = base_model
        
        # Reward head
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
        )
        
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Only train reward head
        for param in self.reward_head.parameters():
            param.requires_grad = True
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute reward score.
        
        Args:
            input_ids: [batch, seq_len]
            attention_mask: Optional attention mask
            
        Returns:
            Reward scores [batch, 1]
        """
        # Get last hidden state
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        hidden_states = outputs.get('last_hidden_state', outputs.get('hidden_states', [None])[-1])
        
        if hidden_states is None:
            # Fallback: use logits
            hidden_states = outputs['logits']
        
        # Get last token representation
        if attention_mask is not None:
            # Find last non-padding token
            seq_lens = attention_mask.sum(dim=1) - 1
            last_hidden = hidden_states[torch.arange(hidden_states.size(0)), seq_lens]
        else:
            last_hidden = hidden_states[:, -1, :]
        
        # Compute reward
        reward = self.reward_head(last_hidden)
        
        return reward
    
    def compute_preference_loss(
        self,
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute preference loss (Bradley-Terry model).
        
        Loss = -log(sigmoid(r_chosen - r_rejected))
        """
        return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()


class RewardModelTrainer:
    """Trainer for reward model."""
    
    def __init__(
        self,
        reward_model: RewardModel,
        config: PosttrainingConfig,
        tokenizer: Any,
    ):
        self.model = reward_model
        self.config = config
        self.tokenizer = tokenizer
        
        self.optimizer = AdamW(
            reward_model.reward_head.parameters(),
            lr=config.reward_learning_rate,
            weight_decay=config.weight_decay,
        )
        
        self.device = next(reward_model.parameters()).device
    
    def train(
        self,
        preference_data: List[PreferencePair],
    ) -> Dict[str, float]:
        """Train reward model on preference data."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Process in batches
        batch_size = 8
        
        for i in range(0, len(preference_data), batch_size):
            batch = preference_data[i:i + batch_size]
            
            chosen_inputs = []
            rejected_inputs = []
            
            for pair in batch:
                # Tokenize chosen
                chosen_text = f"{pair.prompt}\n\n{pair.chosen}"
                chosen_enc = self.tokenizer(
                    chosen_text,
                    max_length=self.config.sft_max_length,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt',
                )
                chosen_inputs.append(chosen_enc['input_ids'])
                
                # Tokenize rejected
                rejected_text = f"{pair.prompt}\n\n{pair.rejected}"
                rejected_enc = self.tokenizer(
                    rejected_text,
                    max_length=self.config.sft_max_length,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt',
                )
                rejected_inputs.append(rejected_enc['input_ids'])
            
            # Stack and move to device
            chosen_ids = torch.cat(chosen_inputs, dim=0).to(self.device)
            rejected_ids = torch.cat(rejected_inputs, dim=0).to(self.device)
            
            # Forward pass
            chosen_rewards = self.model(chosen_ids)
            rejected_rewards = self.model(rejected_ids)
            
            # Compute loss
            loss = self.model.compute_preference_loss(chosen_rewards, rejected_rewards)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            total_loss += loss.item()
            
            # Accuracy
            correct += (chosen_rewards > rejected_rewards).sum().item()
            total += len(batch)
        
        return {
            'reward_loss': total_loss / (len(preference_data) // batch_size),
            'reward_accuracy': correct / total if total > 0 else 0,
        }


# =============================================================================
# PPO Optimizer for RLHF
# =============================================================================

@dataclass
class PPOExperience:
    """Single PPO experience."""
    query_ids: torch.Tensor
    response_ids: torch.Tensor
    old_log_probs: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor


class PPOOptimizer:
    """
    Proximal Policy Optimization for RLHF
    
    Optimizes the policy (LLM) using rewards from the reward model
    while staying close to the reference policy (KL constraint).
    """
    
    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        reward_model: RewardModel,
        config: PosttrainingConfig,
        tokenizer: Any,
    ):
        self.policy = policy_model
        self.ref_policy = ref_model
        self.reward_model = reward_model
        self.config = config
        self.tokenizer = tokenizer
        
        # Freeze reference model
        for param in self.ref_policy.parameters():
            param.requires_grad = False
        
        # Value head (for advantage estimation)
        hidden_size = 4096  # Adjust based on model
        self.value_head = nn.Linear(hidden_size, 1)
        
        # Optimizer
        policy_params = list(self.policy.parameters()) + list(self.value_head.parameters())
        self.optimizer = AdamW(
            policy_params,
            lr=config.ppo_learning_rate,
            weight_decay=config.weight_decay,
        )
        
        self.device = next(policy_model.parameters()).device
        self.value_head = self.value_head.to(self.device)
        
        # Experience buffer
        self.experience_buffer: List[PPOExperience] = []
        
        # KL coefficient (adaptive)
        self.kl_coef = config.kl_coef
    
    def generate_responses(
        self,
        prompts: List[str],
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """
        Generate responses and collect log probabilities.
        
        Returns:
            responses: Generated response texts
            response_ids: Token IDs
            log_probs: Log probabilities of generated tokens
        """
        self.policy.eval()
        
        responses = []
        all_response_ids = []
        all_log_probs = []
        
        with torch.no_grad():
            for prompt in prompts:
                # Tokenize prompt
                prompt_enc = self.tokenizer(
                    prompt,
                    return_tensors='pt',
                ).to(self.device)
                
                prompt_ids = prompt_enc['input_ids']
                
                # Generate
                generated_ids = self._generate(
                    prompt_ids,
                    max_new_tokens=self.config.max_new_tokens,
                )
                
                # Get response (without prompt)
                response_ids = generated_ids[:, prompt_ids.size(1):]
                
                # Compute log probs
                outputs = self.policy(input_ids=generated_ids)
                logits = outputs['logits']
                
                # Shift for next token prediction
                shift_logits = logits[:, prompt_ids.size(1)-1:-1, :]
                log_probs = F.log_softmax(shift_logits, dim=-1)
                
                # Gather log probs for generated tokens
                token_log_probs = log_probs.gather(
                    -1, response_ids.unsqueeze(-1)
                ).squeeze(-1)
                
                # Decode response
                response_text = self.tokenizer.decode(
                    response_ids[0],
                    skip_special_tokens=True,
                )
                
                responses.append(response_text)
                all_response_ids.append(response_ids)
                all_log_probs.append(token_log_probs)
        
        self.policy.train()
        
        return responses, all_response_ids, all_log_probs
    
    def _generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        """Generate tokens autoregressively."""
        generated = prompt_ids
        
        for _ in range(max_new_tokens):
            outputs = self.policy(input_ids=generated)
            logits = outputs['logits'][:, -1, :]
            
            # Sample
            probs = F.softmax(logits / self.config.temperature, dim=-1)
            
            # Top-p sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum - sorted_probs > self.config.top_p
            sorted_probs[mask] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            
            next_token = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_indices.gather(-1, next_token)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        return generated
    
    def compute_rewards(
        self,
        prompts: List[str],
        responses: List[str],
    ) -> torch.Tensor:
        """Compute rewards for prompt-response pairs."""
        self.reward_model.eval()
        
        rewards = []
        
        with torch.no_grad():
            for prompt, response in zip(prompts, responses):
                text = f"{prompt}\n\n{response}"
                
                encoding = self.tokenizer(
                    text,
                    max_length=self.config.sft_max_length,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt',
                ).to(self.device)
                
                reward = self.reward_model(encoding['input_ids'])
                rewards.append(reward)
        
        return torch.cat(rewards, dim=0)
    
    def compute_kl_penalty(
        self,
        policy_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence penalty."""
        kl = policy_log_probs - ref_log_probs
        return self.kl_coef * kl.mean()
    
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE (Generalized Advantage Estimation).
        
        Returns:
            advantages: GAE advantages
            returns: Value targets
        """
        # Simple implementation for single-step rewards
        advantages = rewards - values
        returns = rewards
        
        return advantages, returns
    
    def ppo_step(
        self,
        experiences: List[PPOExperience],
    ) -> Dict[str, float]:
        """
        Execute one PPO update step.
        
        Uses clipped objective to prevent too large policy updates.
        """
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        
        for epoch in range(self.config.ppo_epochs):
            random.shuffle(experiences)
            
            for exp in experiences:
                # Current policy log probs
                outputs = self.policy(input_ids=exp.response_ids)
                logits = outputs['logits'][:, :-1, :]
                current_log_probs = F.log_softmax(logits, dim=-1)
                
                # Gather for actual tokens
                action_log_probs = current_log_probs.gather(
                    -1, exp.response_ids[:, 1:].unsqueeze(-1)
                ).squeeze(-1).mean(dim=-1)
                
                # Policy ratio
                ratio = torch.exp(action_log_probs - exp.old_log_probs.mean(dim=-1))
                
                # Clipped objective
                clipped_ratio = torch.clamp(
                    ratio,
                    1 - self.config.ppo_clip_ratio,
                    1 + self.config.ppo_clip_ratio,
                )
                
                policy_loss = -torch.min(
                    ratio * exp.advantages,
                    clipped_ratio * exp.advantages,
                ).mean()
                
                # Value loss
                values = self.value_head(outputs.get('last_hidden_state', logits)[:, -1, :])
                value_loss = F.mse_loss(values, exp.returns)
                
                # Entropy bonus
                probs = F.softmax(logits, dim=-1)
                entropy = -(probs * current_log_probs).sum(dim=-1).mean()
                
                # Total loss
                loss = (
                    policy_loss +
                    self.config.ppo_vf_coef * value_loss -
                    self.config.ppo_entropy_coef * entropy
                )
                
                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
        
        n = len(experiences) * self.config.ppo_epochs
        
        return {
            'policy_loss': total_policy_loss / n,
            'value_loss': total_value_loss / n,
            'entropy': total_entropy / n,
        }
    
    def train_step(
        self,
        prompts: List[str],
    ) -> Dict[str, float]:
        """
        Complete RLHF training step.
        
        1. Generate responses
        2. Compute rewards
        3. Compute advantages
        4. PPO update
        """
        # Generate responses
        responses, response_ids, old_log_probs = self.generate_responses(prompts)
        
        # Compute rewards
        rewards = self.compute_rewards(prompts, responses)
        
        # Compute reference log probs for KL penalty
        with torch.no_grad():
            ref_outputs = self.ref_policy(input_ids=torch.cat(response_ids, dim=0))
            ref_logits = ref_outputs['logits']
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        
        # Add KL penalty to rewards
        kl_penalty = self.compute_kl_penalty(
            torch.cat(old_log_probs, dim=0).mean(),
            ref_log_probs.mean(),
        )
        adjusted_rewards = rewards - kl_penalty
        
        # Compute values
        with torch.no_grad():
            outputs = self.policy(input_ids=torch.cat(response_ids, dim=0))
            hidden = outputs.get('last_hidden_state', outputs['logits'])[:, -1, :]
            values = self.value_head(hidden)
        
        # Compute advantages
        advantages, returns = self.compute_advantages(adjusted_rewards, values)
        
        # Create experiences
        experiences = []
        for i in range(len(prompts)):
            exp = PPOExperience(
                query_ids=torch.tensor([]),
                response_ids=response_ids[i],
                old_log_probs=old_log_probs[i],
                rewards=rewards[i:i+1],
                advantages=advantages[i:i+1],
                returns=returns[i:i+1],
                values=values[i:i+1],
            )
            experiences.append(exp)
        
        # PPO update
        metrics = self.ppo_step(experiences)
        
        metrics['mean_reward'] = rewards.mean().item()
        metrics['kl_penalty'] = kl_penalty.item()
        
        return metrics


# =============================================================================
# RLHF Trainer
# =============================================================================

class RLHFTrainer:
    """
    Complete RLHF Training Pipeline
    
    Steps:
    1. Train reward model on preference data
    2. Fine-tune policy with PPO using reward model
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: PosttrainingConfig,
        tokenizer: Any,
    ):
        self.config = config
        self.tokenizer = tokenizer
        
        # Create reference model (frozen copy)
        self.ref_model = copy.deepcopy(model)
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # Create reward model
        hidden_size = 4096  # Adjust based on model architecture
        self.reward_model = RewardModel(copy.deepcopy(model), hidden_size)
        
        # Reward model trainer
        self.reward_trainer = RewardModelTrainer(
            self.reward_model,
            config,
            tokenizer,
        )
        
        # PPO optimizer
        self.ppo = PPOOptimizer(
            model,
            self.ref_model,
            self.reward_model,
            config,
            tokenizer,
        )
    
    def train(
        self,
        preference_data: List[PreferencePair],
        prompts: List[str],
        num_ppo_steps: int = 1000,
    ) -> Dict[str, Any]:
        """
        Run complete RLHF training.
        
        Args:
            preference_data: Human preference data for reward model
            prompts: Prompts for PPO training
            num_ppo_steps: Number of PPO training steps
        """
        results = {}
        
        # Step 1: Train reward model
        logger.info("Training reward model...")
        reward_metrics = self.reward_trainer.train(preference_data)
        results['reward_training'] = reward_metrics
        logger.info(f"Reward model: {reward_metrics}")
        
        # Step 2: PPO training
        logger.info("Starting PPO training...")
        
        batch_size = self.config.ppo_batch_size
        ppo_metrics_history = []
        
        for step in range(num_ppo_steps):
            # Sample prompts
            batch_prompts = random.sample(prompts, min(batch_size, len(prompts)))
            
            # PPO step
            metrics = self.ppo.train_step(batch_prompts)
            ppo_metrics_history.append(metrics)
            
            if (step + 1) % 10 == 0:
                avg_reward = sum(m['mean_reward'] for m in ppo_metrics_history[-10:]) / 10
                logger.info(f"PPO Step {step + 1}/{num_ppo_steps}, Avg Reward: {avg_reward:.4f}")
        
        results['ppo_training'] = {
            'final_metrics': ppo_metrics_history[-1],
            'num_steps': num_ppo_steps,
        }
        
        return results


# =============================================================================
# Direct Preference Optimization (DPO)
# =============================================================================

class DPOTrainer:
    """
    Direct Preference Optimization
    
    Alternative to RLHF that directly optimizes on preference data
    without training a separate reward model.
    
    Loss: -log(sigmoid(beta * (log(pi(y_w|x)/pi_ref(y_w|x)) - log(pi(y_l|x)/pi_ref(y_l|x)))))
    """
    
    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        config: PosttrainingConfig,
        tokenizer: Any,
    ):
        self.model = model
        self.ref_model = ref_model
        self.config = config
        self.tokenizer = tokenizer
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.dpo_learning_rate,
            weight_decay=config.weight_decay,
        )
        
        self.device = next(model.parameters()).device
    
    def compute_log_probs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probabilities for a sequence."""
        outputs = model(input_ids=input_ids)
        logits = outputs['logits']
        
        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Log softmax
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather log probs for actual tokens
        token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        
        # Sum log probs (ignore padding)
        mask = (shift_labels != -100).float()
        sequence_log_probs = (token_log_probs * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
        
        return sequence_log_probs
    
    def dpo_loss(
        self,
        policy_chosen_log_probs: torch.Tensor,
        policy_rejected_log_probs: torch.Tensor,
        ref_chosen_log_probs: torch.Tensor,
        ref_rejected_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute DPO loss.
        
        Loss = -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))
        """
        log_ratio_chosen = policy_chosen_log_probs - ref_chosen_log_probs
        log_ratio_rejected = policy_rejected_log_probs - ref_rejected_log_probs
        
        logits = self.config.dpo_beta * (log_ratio_chosen - log_ratio_rejected)
        
        loss = -F.logsigmoid(logits).mean()
        
        return loss
    
    def train(
        self,
        preference_data: List[PreferencePair],
    ) -> Dict[str, float]:
        """Train with DPO."""
        self.model.train()
        
        total_loss = 0.0
        chosen_rewards = []
        rejected_rewards = []
        
        batch_size = 4
        
        for epoch in range(self.config.dpo_epochs):
            random.shuffle(preference_data)
            
            for i in range(0, len(preference_data), batch_size):
                batch = preference_data[i:i + batch_size]
                
                # Tokenize chosen and rejected
                chosen_encodings = []
                rejected_encodings = []
                
                for pair in batch:
                    chosen_text = f"{pair.prompt}\n\n{pair.chosen}"
                    rejected_text = f"{pair.prompt}\n\n{pair.rejected}"
                    
                    chosen_enc = self.tokenizer(
                        chosen_text,
                        max_length=self.config.sft_max_length,
                        truncation=True,
                        padding='max_length',
                        return_tensors='pt',
                    )
                    rejected_enc = self.tokenizer(
                        rejected_text,
                        max_length=self.config.sft_max_length,
                        truncation=True,
                        padding='max_length',
                        return_tensors='pt',
                    )
                    
                    chosen_encodings.append(chosen_enc['input_ids'])
                    rejected_encodings.append(rejected_enc['input_ids'])
                
                chosen_ids = torch.cat(chosen_encodings, dim=0).to(self.device)
                rejected_ids = torch.cat(rejected_encodings, dim=0).to(self.device)
                
                # Compute log probs
                policy_chosen_log_probs = self.compute_log_probs(
                    self.model, chosen_ids, chosen_ids
                )
                policy_rejected_log_probs = self.compute_log_probs(
                    self.model, rejected_ids, rejected_ids
                )
                
                with torch.no_grad():
                    ref_chosen_log_probs = self.compute_log_probs(
                        self.ref_model, chosen_ids, chosen_ids
                    )
                    ref_rejected_log_probs = self.compute_log_probs(
                        self.ref_model, rejected_ids, rejected_ids
                    )
                
                # DPO loss
                loss = self.dpo_loss(
                    policy_chosen_log_probs,
                    policy_rejected_log_probs,
                    ref_chosen_log_probs,
                    ref_rejected_log_probs,
                )
                
                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                total_loss += loss.item()
                
                # Track implicit rewards
                with torch.no_grad():
                    chosen_reward = self.config.dpo_beta * (
                        policy_chosen_log_probs - ref_chosen_log_probs
                    )
                    rejected_reward = self.config.dpo_beta * (
                        policy_rejected_log_probs - ref_rejected_log_probs
                    )
                    chosen_rewards.extend(chosen_reward.tolist())
                    rejected_rewards.extend(rejected_reward.tolist())
            
            logger.info(f"DPO Epoch {epoch + 1}/{self.config.dpo_epochs}")
        
        return {
            'dpo_loss': total_loss / (len(preference_data) // batch_size * self.config.dpo_epochs),
            'chosen_reward': sum(chosen_rewards) / len(chosen_rewards) if chosen_rewards else 0,
            'rejected_reward': sum(rejected_rewards) / len(rejected_rewards) if rejected_rewards else 0,
            'reward_margin': (sum(chosen_rewards) - sum(rejected_rewards)) / len(chosen_rewards) if chosen_rewards else 0,
        }


# =============================================================================
# Constitutional AI
# =============================================================================

@dataclass
class ConstitutionalPrinciple:
    """A single constitutional principle."""
    name: str
    description: str
    critique_prompt: str
    revision_prompt: str


class ConstitutionalAI:
    """
    Constitutional AI Implementation
    
    Ensures model outputs align with specified principles
    through critique and revision cycles.
    """
    
    DEFAULT_PRINCIPLES = [
        ConstitutionalPrinciple(
            name="helpful",
            description="Responses should be helpful and informative",
            critique_prompt="Is this response helpful and informative? What could be improved?",
            revision_prompt="Revise the response to be more helpful and informative while maintaining accuracy.",
        ),
        ConstitutionalPrinciple(
            name="harmless",
            description="Responses should not cause harm",
            critique_prompt="Could this response potentially cause harm? What harmful content should be removed?",
            revision_prompt="Revise the response to remove any potentially harmful content.",
        ),
        ConstitutionalPrinciple(
            name="honest",
            description="Responses should be truthful and acknowledge uncertainty",
            critique_prompt="Is this response honest and truthful? Does it acknowledge uncertainty appropriately?",
            revision_prompt="Revise the response to be more honest and acknowledge any uncertainties.",
        ),
        ConstitutionalPrinciple(
            name="ethical",
            description="Responses should follow ethical guidelines",
            critique_prompt="Does this response follow ethical guidelines? What ethical issues are present?",
            revision_prompt="Revise the response to address any ethical concerns.",
        ),
    ]
    
    def __init__(
        self,
        model: nn.Module,
        config: PosttrainingConfig,
        tokenizer: Any,
        principles: Optional[List[ConstitutionalPrinciple]] = None,
    ):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.principles = principles or self.DEFAULT_PRINCIPLES
        
        self.device = next(model.parameters()).device
    
    def critique(
        self,
        prompt: str,
        response: str,
        principle: ConstitutionalPrinciple,
    ) -> str:
        """Generate critique based on principle."""
        critique_input = f"""
Original prompt: {prompt}

Response to critique: {response}

Principle: {principle.description}

{principle.critique_prompt}

Critique:"""
        
        return self._generate(critique_input)
    
    def revise(
        self,
        prompt: str,
        response: str,
        critique: str,
        principle: ConstitutionalPrinciple,
    ) -> str:
        """Revise response based on critique."""
        revision_input = f"""
Original prompt: {prompt}

Original response: {response}

Critique: {critique}

{principle.revision_prompt}

Revised response:"""
        
        return self._generate(revision_input)
    
    def _generate(self, prompt: str) -> str:
        """Generate response from model."""
        self.model.eval()
        
        encoding = self.tokenizer(
            prompt,
            return_tensors='pt',
        ).to(self.device)
        
        with torch.no_grad():
            # Simple generation
            generated = encoding['input_ids']
            
            for _ in range(self.config.max_new_tokens):
                outputs = self.model(input_ids=generated)
                logits = outputs['logits'][:, -1, :]
                
                probs = F.softmax(logits / self.config.temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        response = self.tokenizer.decode(
            generated[0, encoding['input_ids'].size(1):],
            skip_special_tokens=True,
        )
        
        self.model.train()
        
        return response
    
    def apply_constitution(
        self,
        prompt: str,
        initial_response: str,
        max_iterations: int = 2,
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Apply constitutional AI process.
        
        Returns:
            final_response: Revised response
            history: List of critique/revision pairs
        """
        history = []
        current_response = initial_response
        
        for iteration in range(max_iterations):
            for principle in self.principles:
                # Critique
                critique = self.critique(prompt, current_response, principle)
                
                # Revise
                revised = self.revise(prompt, current_response, critique, principle)
                
                history.append({
                    'iteration': iteration,
                    'principle': principle.name,
                    'critique': critique,
                    'original': current_response,
                    'revised': revised,
                })
                
                current_response = revised
        
        return current_response, history
    
    def generate_training_data(
        self,
        prompts: List[str],
        num_samples: int = 1000,
    ) -> List[PreferencePair]:
        """
        Generate constitutional AI training data.
        
        Creates preference pairs where revised responses are preferred.
        """
        preference_data = []
        
        for prompt in prompts[:num_samples]:
            # Generate initial response
            initial = self._generate(prompt)
            
            # Apply constitution
            revised, _ = self.apply_constitution(prompt, initial)
            
            # Create preference pair (revised is chosen)
            pair = PreferencePair(
                prompt=prompt,
                chosen=revised,
                rejected=initial,
                metadata={'method': 'constitutional'},
            )
            
            preference_data.append(pair)
        
        return preference_data


# =============================================================================
# Value Aligner (Combines all methods)
# =============================================================================

class ValueAligner:
    """
    Unified value alignment interface.
    
    Supports:
    - SFT
    - RLHF
    - DPO
    - Constitutional AI
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: PosttrainingConfig,
        tokenizer: Any,
    ):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        
        # Initialize trainers based on method
        self.sft_trainer = SFTTrainer(model, config, tokenizer)
        
        if config.method == AlignmentMethod.RLHF:
            self.rlhf_trainer = RLHFTrainer(model, config, tokenizer)
        
        if config.method == AlignmentMethod.DPO:
            ref_model = copy.deepcopy(model)
            for param in ref_model.parameters():
                param.requires_grad = False
            self.dpo_trainer = DPOTrainer(model, ref_model, config, tokenizer)
        
        if config.method == AlignmentMethod.CONSTITUTIONAL:
            self.constitutional = ConstitutionalAI(model, config, tokenizer)
    
    def align(
        self,
        sft_data: Optional[List[Conversation]] = None,
        preference_data: Optional[List[PreferencePair]] = None,
        prompts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run full alignment pipeline.
        
        Args:
            sft_data: Supervised fine-tuning data
            preference_data: Human preference data
            prompts: Prompts for RL training
        """
        results = {}
        
        # Step 1: SFT (always first if data provided)
        if sft_data:
            logger.info("Running SFT...")
            sft_results = self.sft_trainer.train(sft_data)
            results['sft'] = sft_results
        
        # Step 2: Alignment method specific training
        if self.config.method == AlignmentMethod.RLHF:
            if preference_data and prompts:
                logger.info("Running RLHF...")
                rlhf_results = self.rlhf_trainer.train(
                    preference_data,
                    prompts,
                )
                results['rlhf'] = rlhf_results
        
        elif self.config.method == AlignmentMethod.DPO:
            if preference_data:
                logger.info("Running DPO...")
                dpo_results = self.dpo_trainer.train(preference_data)
                results['dpo'] = dpo_results
        
        elif self.config.method == AlignmentMethod.CONSTITUTIONAL:
            if prompts:
                logger.info("Running Constitutional AI...")
                # Generate preference data
                const_data = self.constitutional.generate_training_data(prompts)
                
                # Train with DPO on constitutional data
                ref_model = copy.deepcopy(self.model)
                dpo_trainer = DPOTrainer(
                    self.model, ref_model, self.config, self.tokenizer
                )
                const_results = dpo_trainer.train(const_data)
                results['constitutional'] = const_results
        
        return results
