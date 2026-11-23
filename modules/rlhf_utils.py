"""
RLHF Utils Module for Streamlit Integration
This module provides the run_rlhf_training function for Streamlit app
Updated: December 2025 with PPO epochs implementation
"""
import os
# Set CUDA environment variables for better multi-GPU performance
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Use first two GPUs

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import numpy as np
from typing import Dict, List, Any
import logging
from datetime import datetime
from dataclasses import dataclass
import streamlit as st
import json


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExpertRating:
    """Structure for expert ratings from Streamlit"""
    section_prompt: str
    content: str
    rating: float  # 0.0-1.0 scale
    timestamp: str

class StreamlitRatingDataset(Dataset):
    """Dataset for training reward model on Streamlit ratings"""
    
    def __init__(self, ratings: List[ExpertRating], tokenizer, max_length=512):
        self.ratings = ratings
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        rating = self.ratings[idx]
        
        # Combine prompt and content as input
        text = f"Prompt: {rating.section_prompt}\nContent: {rating.content}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Single rating target (0.0-1.0)
        reward = torch.tensor(rating.rating, dtype=torch.float32)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'reward': reward
        }

class SimpleRewardModel(nn.Module):
    """Enhanced reward model with layer normalization for stability"""
    
    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Freeze most of the encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Optionally unfreeze last few layers for better adaptation
        try:
            if hasattr(self.encoder, 'encoder') and hasattr(self.encoder.encoder, 'layer'):
                for param in self.encoder.encoder.layer[-2:].parameters():
                    param.requires_grad = True
        except AttributeError:
            logger.info("Using fully frozen encoder (architecture-specific unfreezing not available)")
            
        hidden_size = self.encoder.config.hidden_size
        
        # Enhanced reward head with layer normalization
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output between 0-1
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        reward = self.reward_head(pooled_output).squeeze(-1)
        return reward

class StreamlitRLHFTrainer:
    """RLHF Trainer with proper PPO epochs implementation"""
    
    def __init__(self, model_path: str, device='auto', ppo_epochs: int = 4, num_trainable_layers: int = 2):
        # Enhanced device handling
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, int):
            self.device = torch.device(f'cuda:{device}')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and models
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load policy model (the main generation model)
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map='auto',
            trust_remote_code=True
        )
        
        # Store model path for reference model creation
        self.model_path = model_path

        # ---- FREEZE POLICY LAYERS HERE ----
        self.freeze_policy_layers(num_trainable_layers=num_trainable_layers)

        # Create reward model
        self.reward_model = SimpleRewardModel('meta-llama/Llama-3.2-3B-Instruct')
        
        # Move reward model to device
        if not isinstance(self.device, str) or self.device != 'auto':
            self.reward_model.to(self.device)
        
        # Optimizers
        self.reward_optimizer = optim.AdamW(self.reward_model.parameters(), lr=1e-5)
        self.policy_optimizer = optim.AdamW(self.policy_model.parameters(), lr=5e-7)
        
        # PPO parameters
        self.ppo_epochs = ppo_epochs  # Now configurable
        self.clip_epsilon = 0.1
        self.kl_coef = 0.1
        
        logger.info(f"PPO configuration: epochs={self.ppo_epochs}, clip_epsilon={self.clip_epsilon}, kl_coef={self.kl_coef}")

    def freeze_policy_layers(self, num_trainable_layers: int = 2):
        """
        Freeze all transformer blocks except the last `num_trainable_layers`.
        Adjusted to work with common HF architectures (LLaMA / GPT2-style).
        """
        model = self.policy_model

        layers = None

        # LLaMA / Mistral / Llama-3 style
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers

        # GPT-2 / GPT-Neo style
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            layers = model.transformer.h

        if layers is None:
            logger.warning("Could not detect transformer layers to freeze; skipping layer freezing.")
            return

        n_layers = len(layers)
        n_train = min(num_trainable_layers, n_layers)

        # Freeze all but last n_train blocks
        for layer in layers[: n_layers - n_train]:
            for p in layer.parameters():
                p.requires_grad = False

        # Optionally freeze embeddings too (often helpful)
        if hasattr(model, "get_input_embeddings"):
            for p in model.get_input_embeddings().parameters():
                p.requires_grad = False

        # Usually we want the lm_head to stay trainable
        if hasattr(model, "lm_head"):
            for p in model.lm_head.parameters():
                p.requires_grad = True

        # Log how many params are trainable
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Policy params: {trainable}/{total} trainable "
                    f"({trainable/total:.2%}); last {n_train} layers unfrozen.")

    def prepare_ratings_data(self, ratings_dict: Dict[str, float], 
                           output_dict: Dict[str, str]) -> List[ExpertRating]:
        """Convert Streamlit ratings to ExpertRating objects"""
        ratings_list = []
        timestamp = datetime.now().isoformat()
        
        for title, rating in ratings_dict.items():
            content = None
            if title.rstrip() in output_dict:
                content = output_dict[title.rstrip()]
            else:
                for key in output_dict.keys():
                    if title.rstrip() in key or key.rstrip() == title.rstrip():
                        content = output_dict[key]
                        break

            if content is not None:
                expert_rating = ExpertRating(
                    section_prompt=st.session_state.prmpt[title.rstrip()],
                    content=content,
                    rating=rating,
                    timestamp=timestamp
                )
                ratings_list.append(expert_rating)
            else:
                logger.warning(f"Could not find content for title: {title}")
        
        logger.info(f"Prepared {len(ratings_list)} ratings for training")
        return ratings_list
    
    def train_reward_model_quick(self, ratings: List[ExpertRating], 
                               epochs: int = 5, 
                               batch_size: int = 4,
                               accumulation_steps: int = 4):
        """Enhanced reward model training with gradient accumulation"""
        if len(ratings) == 0:
            logger.warning("No ratings provided for training")
            return
        
        dataset = StreamlitRatingDataset(ratings, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.reward_model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            self.reward_optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                target_rewards = batch['reward'].to(self.device)
                
                # Forward pass
                predicted_rewards = self.reward_model(input_ids, attention_mask)
                
                # MSE Loss scaled by accumulation steps
                loss = nn.MSELoss()(predicted_rewards, target_rewards)
                loss = loss / accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Update weights after accumulation steps
                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(), 1.0)
                    self.reward_optimizer.step()
                    self.reward_optimizer.zero_grad()
                
                total_loss += loss.item() * accumulation_steps
                num_batches += 1
            
            # Final update if there are remaining gradients
            if num_batches % accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(), 1.0)
                self.reward_optimizer.step()
                self.reward_optimizer.zero_grad()
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            logger.info(f"Reward model epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    
    def generate_and_score_samples(self, content_dict: Dict[str, str], 
                                 num_samples: int = 2) -> List[Dict]:
        """Generate variations and score them with improved memory management"""
        samples = []
        
        self.policy_model.eval()
        self.reward_model.eval()
        
        # Clear GPU cache before generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        for title, content in content_dict.items():
            # Create prompt for generation
            prompt = f"Improve this policy section:\nTitle: {title}\nOriginal: {content}\nImproved:"
            
            inputs = self.tokenizer(
                prompt,
                return_tensors='pt',
                truncation=True,
                max_length=512
            )
            
            # Move to appropriate device
            if hasattr(self.policy_model, 'device'):
                inputs = {k: v.to(self.policy_model.device) for k, v in inputs.items()}
            else:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate samples
            with torch.no_grad():
                for _ in range(num_samples):
                    output = self.policy_model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id,
                        no_repeat_ngram_size=2
                    )
                    
                    generated_text = self.tokenizer.decode(
                        output[0][inputs['input_ids'].shape[1]:],
                        skip_special_tokens=True
                    )
                    
                    # Score the generated text
                    score_input = f"Title: {title}\nContent: {generated_text}"
                    score_tokens = self.tokenizer(
                        score_input,
                        return_tensors='pt',
                        truncation=True,
                        max_length=512
                    ).to(self.device)
                    
                    reward_score = self.reward_model(
                        score_tokens['input_ids'],
                        score_tokens['attention_mask']
                    ).item()
                    
                    samples.append({
                        'title': title,
                        'original_content': content,
                        'generated_content': generated_text,
                        'reward_score': reward_score,
                        'prompt': prompt,
                        'full_generated_tokens': output[0],
                        'input_length': inputs['input_ids'].shape[1]
                    })
                    
                    # Clear cache after each generation
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        return samples
    
    def ppo_update_with_epochs(self, samples: List[Dict], use_kl_penalty: bool = True):
        """
        PPO update with proper epoch iteration over rollout data
        This is the correct implementation following PPO algorithm
        """
        logger.info(f'Training policy model with PPO ({self.ppo_epochs} epochs)...')
        
        # Create reference model for KL computation if enabled
        ref_model = None
        if use_kl_penalty:
            try:
                ref_model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map={'':'cpu'},
                    trust_remote_code=True
                )
                ref_model.eval()
                logger.info("Reference model created for KL penalty")
            except Exception as e:
                logger.warning(f"Could not create reference model: {e}")
                use_kl_penalty = False
        
        # Sort samples by reward and keep top performers
        samples.sort(key=lambda x: x['reward_score'], reverse=True)
        top_samples = samples[:max(1, len(samples)//2)]
        
        # Store old log probabilities for PPO clipping
        old_log_probs = []
        self.policy_model.eval()
        
        with torch.no_grad():
            for sample in top_samples:
                full_tokens = sample['full_generated_tokens']
                input_ids = full_tokens.unsqueeze(0) if full_tokens.dim() == 1 else full_tokens
                
                if hasattr(self.policy_model, 'device'):
                    input_ids = input_ids.to(self.policy_model.device)
                else:
                    input_ids = input_ids.to(self.device)
                
                attention_mask = torch.ones_like(input_ids)
                
                # Get old policy log probs
                outputs = self.policy_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Calculate log probabilities for generated tokens
                logits = outputs.logits[:, sample['input_length']-1:-1, :]
                tokens = input_ids[:, sample['input_length']:]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                token_log_probs = torch.gather(log_probs, 2, tokens.unsqueeze(-1)).squeeze(-1)
                old_log_probs.append(token_log_probs.mean().item())
        
        # PPO EPOCHS LOOP - This is the key implementation
        for ppo_epoch in range(self.ppo_epochs):
            self.policy_model.train()
            epoch_loss = 0
            epoch_kl = 0
            epoch_clip_frac = 0
            num_updates = 0
            
            for idx, sample in enumerate(top_samples):
                try:
                    full_tokens = sample['full_generated_tokens']
                    input_length = sample['input_length']
                    
                    input_ids = full_tokens.unsqueeze(0) if full_tokens.dim() == 1 else full_tokens
                    
                    if hasattr(self.policy_model, 'device'):
                        input_ids = input_ids.to(self.policy_model.device)
                    else:
                        input_ids = input_ids.to(self.device)
                    
                    attention_mask = torch.ones_like(input_ids)
                    labels = input_ids.clone()
                    labels[:, :input_length] = -100
                    
                    # Current policy output
                    outputs = self.policy_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    # Calculate current log probabilities
                    logits = outputs.logits[:, input_length-1:-1, :]
                    tokens = input_ids[:, input_length:]
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    token_log_probs = torch.gather(log_probs, 2, tokens.unsqueeze(-1)).squeeze(-1)
                    current_log_prob = token_log_probs.mean()
                    
                    # PPO clipping ratio
                    ratio = torch.exp(current_log_prob - old_log_probs[idx])
                    
                    # Advantage (using reward as simplified advantage)
                    advantage = sample['reward_score']
                    
                    # PPO clipped objective
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage
                    policy_loss = -torch.min(surr1, surr2)
                    
                    # Track clipping fraction
                    clip_frac = ((ratio < 1 - self.clip_epsilon) | (ratio > 1 + self.clip_epsilon)).float().mean()
                    epoch_clip_frac += clip_frac.item()
                    
                    # KL divergence penalty
                    kl_div = torch.tensor(0.0, device=policy_loss.device)
                    if use_kl_penalty and ref_model is not None:
                        try:
                            # 1) Get devices
                            policy_device = outputs.logits.device
                            ref_device = next(ref_model.parameters()).device
                    
                            # 2) Send inputs separately for each model
                            #    (you already have input_ids & attention_mask on policy_device)
                            ref_input_ids = input_ids.to(ref_device)
                            ref_attention_mask = attention_mask.to(ref_device)
                    
                            with torch.no_grad():
                                ref_outputs = ref_model(
                                    input_ids=ref_input_ids,
                                    attention_mask=ref_attention_mask
                                )
                    
                            # 3) Restrict to generated tokens if you want (same as your logic)
                            logits_policy = outputs.logits[:, input_length-1:-1, :]
                            logits_ref = ref_outputs.logits[:, input_length-1:-1, :]
                    
                            # 4) Move ref logits to the same device as policy logits
                            logits_ref = logits_ref.to(policy_device)
                    
                            # 5) Compute KL on same device
                            log_probs_policy = F.log_softmax(logits_policy, dim=-1)
                            log_probs_ref = F.log_softmax(logits_ref, dim=-1)
                    
                            kl_div = F.kl_div(
                                log_probs_policy,
                                log_probs_ref,
                                reduction='batchmean',
                                log_target=True
                            )
                    
                            epoch_kl += kl_div.item()
                    
                        except Exception as e:
                            logger.warning(f"KL computation failed: {e}")
                            kl_div = torch.tensor(0.0, device=policy_loss.device)
                    
                    # Total loss
                    total_loss = policy_loss + (self.kl_coef * kl_div) + (0.1 * outputs.loss)
                    
                    # Backward pass
                    self.policy_optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
                    self.policy_optimizer.step()
                    
                    epoch_loss += total_loss.item()
                    num_updates += 1
                    
                except Exception as e:
                    logger.warning(f"Skipping sample in PPO epoch {ppo_epoch+1}: {str(e)}")
                    continue
            
            # Log epoch statistics
            if num_updates > 0:
                avg_loss = epoch_loss / num_updates
                avg_kl = epoch_kl / num_updates if use_kl_penalty else 0
                avg_clip_frac = epoch_clip_frac / num_updates
                logger.info(
                    f"PPO Epoch {ppo_epoch+1}/{self.ppo_epochs} - "
                    f"Loss: {avg_loss:.4f}, KL: {avg_kl:.4f}, Clip Frac: {avg_clip_frac:.3f}"
                )
        
        # Clean up reference model
        if ref_model is not None:
            del ref_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Save model after all PPO epochs
        if num_updates > 0:
            save_path = f'./fine_tuned_models/model_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            os.makedirs(save_path, exist_ok=True)
            
            self.policy_model.resize_token_embeddings(len(self.tokenizer))
            self.policy_model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            st.session_state.rl_model = save_path
            
            logger.info(f"Model saved to {save_path}")
            logger.info(f"PPO training completed with {self.ppo_epochs} epochs")
        else:
            logger.warning("No PPO updates completed")
    
    def simple_policy_update(self, samples: List[Dict]):
        """Simplified policy update (kept for backward compatibility)"""
        logger.info('Training policy model (simple mode)...')
        self.policy_model.train()
        
        samples.sort(key=lambda x: x['reward_score'], reverse=True)
        top_samples = samples[:max(1, len(samples)//2)]
        
        total_loss = 0
        num_updates = 0
        
        for sample in top_samples:
            try:
                full_tokens = sample['full_generated_tokens']
                input_length = sample['input_length']
                
                input_ids = full_tokens.unsqueeze(0) if full_tokens.dim() == 1 else full_tokens
                
                if hasattr(self.policy_model, 'device'):
                    input_ids = input_ids.to(self.policy_model.device)
                else:
                    input_ids = input_ids.to(self.device)
                
                attention_mask = torch.ones_like(input_ids)
                labels = input_ids.clone()
                labels[:, :input_length] = -100
                
                outputs = self.policy_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                weighted_loss = outputs.loss * sample['reward_score']
                
                self.policy_optimizer.zero_grad()
                weighted_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
                self.policy_optimizer.step()
                
                total_loss += weighted_loss.item()
                num_updates += 1
                
            except Exception as e:
                logger.warning(f"Skipping sample due to error: {str(e)}")
                continue
        
        if num_updates > 0:
            avg_loss = total_loss / num_updates
            
            save_path = f'./fine_tuned_models/model_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            os.makedirs(save_path, exist_ok=True)
            
            self.policy_model.resize_token_embeddings(len(self.tokenizer))
            self.policy_model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            st.session_state.model = save_path
            
            logger.info(f"Model saved to {save_path}")
            logger.info(f"Policy update completed - Average loss: {avg_loss:.4f}")
        else:
            logger.warning("No policy updates completed")
    
    def save_training_data(self, ratings: List[ExpertRating], 
                          save_path: str = "training_data.json"):
        """Save training data for future use"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'ratings': [
                {
                    'section_prompt': r.section_prompt,
                    'content': r.content,
                    'rating': r.rating,
                    'timestamp': r.timestamp
                } for r in ratings
            ]
        }
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Training data saved to {save_path}")

def run_rlhf_training(ratings_dict: Dict[str, float], 
                     output_dict: Dict[str, str], 
                     model_path: str,
                     reward_epochs: int = 5,
                     num_samples: int = 2,
                     ppo_epochs: int = 4,
                     use_ppo_epochs: bool = True,
                     use_kl_penalty: bool = True) -> str:
    """
    Main function called from Streamlit app with PPO epochs support
    
    Args:
        ratings_dict: Dictionary mapping section titles to ratings (0.0-1.0)
        output_dict: Dictionary mapping section titles to generated content
        model_path: Path to the base model
        reward_epochs: Number of epochs for reward model training
        num_samples: Number of samples to generate per content section
        ppo_epochs: Number of PPO epochs (2-10 recommended)
        use_ppo_epochs: Whether to use proper PPO epoch loop
        use_kl_penalty: Whether to use KL penalty in policy update
    
    Returns:
        Status message
    """
    try:
        logger.info("Starting RLHF training with PPO...")
        
        # Initialize trainer with PPO epochs
        if 'model' not in st.session_state:
            current_model_path = model_path
        else:
            current_model_path = st.session_state.model
        
        trainer = StreamlitRLHFTrainer(current_model_path, device='auto', ppo_epochs=ppo_epochs, num_trainable_layers=3)
        
        # Prepare ratings data
        ratings = trainer.prepare_ratings_data(ratings_dict, output_dict)
        
        if len(ratings) == 0:
            return "No valid ratings found for training"
        
        # Save training data
        trainer.save_training_data(ratings)
        
        # Train reward model
        logger.info("Training reward model...")
        trainer.train_reward_model_quick(ratings, epochs=reward_epochs)
        
        # Generate and score samples
        logger.info("Generating and scoring samples...")
        samples = trainer.generate_and_score_samples(output_dict, num_samples=num_samples)
        
        if len(samples) == 0:
            return "No samples generated for policy update"
        
        # Update policy with PPO epochs
        logger.info("Updating policy with PPO...")
        try:
            if use_ppo_epochs:
                trainer.ppo_update_with_epochs(samples, use_kl_penalty=use_kl_penalty)
            else:
                trainer.simple_policy_update(samples)
        except Exception as e:
            logger.warning(f"PPO update failed: {e}, trying simple update...")
            trainer.simple_policy_update(samples)
        
        # Log results
        avg_reward = np.mean([s['reward_score'] for s in samples])
        logger.info(f"Training completed. Average reward score: {avg_reward:.3f}")
        
        return f"RLHF training completed successfully! Average reward: {avg_reward:.3f}, PPO epochs: {ppo_epochs}"
        
    except Exception as e:
        logger.error(f"Error in RLHF training: {str(e)}")
        return f"Training failed: {str(e)}"

# Additional utility functions

def get_training_history(data_file: str = "training_data.json") -> List[Dict]:
    """Get historical training data"""
    if not os.path.exists(data_file):
        return []
    
    try:
        with open(data_file, 'r') as f:
            data = json.load(f)
        return data.get('ratings', [])
    except Exception as e:
        logger.warning(f"Could not load training history: {e}")
        return []

def analyze_rating_patterns(ratings: List[Dict]) -> Dict[str, Any]:
    """Analyze rating patterns for insights"""
    if not ratings:
        return {}
    
    rating_values = [r['rating'] for r in ratings]
    
    return {
        'total_ratings': len(ratings),
        'average_rating': np.mean(rating_values),
        'min_rating': np.min(rating_values),
        'max_rating': np.max(rating_values),
        'std_rating': np.std(rating_values),
        'section_counts': {}
    }
import gc
import torch

def unload_rag():
    """Offload and delete RAG models so RLHF can use the GPU."""
    if "rag" not in st.session_state:
        return

    rag = st.session_state.rag

    # 1) Offload LLM used for RAG
    llm = rag.get("llm")
    llm_model = rag.get("llm_model")

    # Case 1: llm is a raw HF model
    if llm is not None and hasattr(llm, "to"):
        try:
            llm.to("cpu")
        except Exception:
            pass

    # Case 2: LanguageModel wrapper that has a .model attribute
    if llm_model is not None:
        base = getattr(llm_model, "model", None)
        if base is not None and hasattr(base, "to"):
            try:
                base.to("cpu")
            except Exception:
                pass

    # 2) Offload embedding model if it lives on GPU
    embedder = rag.get("embedder")
    if embedder is not None:
        emb_model = getattr(embedder, "model", None)
        if emb_model is not None and hasattr(emb_model, "to"):
            try:
                emb_model.to("cpu")
            except Exception:
                pass

    # 3) Delete all RAG objects so Python can free them
    for key in list(rag.keys()):
        rag[key] = None
    del st.session_state.rag

    # 4) Force memory cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def start_rlhf(ratings_dict: Dict[str, float], 
                     output_dict: Dict[str, str], 
                     model_path: str,
                     reward_epochs: int = 5,
                     num_samples: int = 2,
                     ppo_epochs: int = 4,
                     use_ppo_epochs: bool = True,
                     use_kl_penalty: bool = True):
    # 1) Offload or delete the inference model
    unload_rag()

    # 2) Run training
    status = run_rlhf_training(ratings_dict, output_dict, model_path, reward_epochs, num_samples)
    st.write(status)
