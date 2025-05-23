import os
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from typing import Dict, Any
from torch.optim.lr_scheduler import _LRScheduler

from torch.cuda.amp import GradScaler, autocast

from src.model import LLaMAModel


def train_model(
    model: LLaMAModel,
    train_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    criterion: nn.Module,
    training_config: Dict[str, Any],
    device: torch.device
) -> None:
    """
    Train the LLaMA model.
    
    Args:
        model: LLaMA model to train
        train_loader: DataLoader for training data
        optimizer: Optimizer for updating model parameters
        scheduler: Learning rate scheduler
        criterion: Loss function
        training_config: Training configuration dictionary
        device: Device to train on (cuda or cpu)
    """
    num_epochs = training_config['num_epochs']
    save_freq = training_config['save_freq']
    log_freq = training_config['log_freq']
    save_dir = training_config.get("save_dir", "checkpoints")

    wandb.init(project="llama_training", config=training_config)
    wandb.watch(model, log="all", log_freq=log_freq)

    model.to(device)
    model.train()

    os.makedirs(save_dir, exist_ok=True)
    global_step = 0
    total_loss = 0

    scaler = GradScaler()

    for epoch in range(1, num_epochs + 1):
        for batch in train_loader:
            if isinstance(batch, dict):
                input_tokens = batch['input_ids'].to(device)
            else:
                input_tokens = batch.to(device)

            # Shift tokens for language modeling task (predict next token)
            target_tokens = input_tokens[:, 1:].contiguous()
            input_tokens = input_tokens[:, :-1].contiguous()

            optimizer.zero_grad()

            # Use mixed precision training
            with autocast():
                logits, _ = model(input_tokens, scaling_factor=1.0)
                loss = criterion(
                    logits.view(-1, logits.size(-1)),
                    target_tokens.view(-1)
                )

            # Scale gradients and backpropagate
            scaler.scale(loss).backward()

            # Unscale before gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update weights
            scaler.step(optimizer)
            scaler.update()

            # Update learning rate
            scheduler.step()
            total_loss += loss.item()
            global_step += 1

            # Log metrics
            if global_step % log_freq == 0:
                avg_loss = total_loss / log_freq
                wandb.log({
                    "step": global_step,
                    "train_loss": avg_loss,
                    "batch_size": training_config["batch_size"],
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "epoch": epoch
                })
                total_loss = 0

            # Save checkpoint
            if global_step % save_freq == 0:
                checkpoint_path = os.path.join(save_dir, f"llama_checkpoint_step_{global_step}.pt")
                torch.save(model.state_dict(), checkpoint_path)
                wandb.save(checkpoint_path)

    wandb.finish() 