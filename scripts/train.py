import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup

from src.configs import LLaMAConfig, get_default_training_config
from src.training import train_model
from src.data import load_tokenizer, load_data, save_dataset
from src.model import LLaMAModel


def main():
    model_config = LLaMAConfig(
        vocab_size=32000,           
        hidden_size=256,           
        num_hidden_layers=6,        
        num_attention_heads=8,      
        intermediate_size=1024,     
        rms_norm_eps=1e-6,
        max_position_embeddings=1024, 
        initializer_range=0.02,
        pad_token_id=0              
    )

    training_config = get_default_training_config()
    
    print(f"Saving dataset to '{training_config['save_dir']}'")
    save_dataset('data')

    train_dataloader = load_data('data')
    print(f'Dataloader length: {len(train_dataloader)}')
    
    tokenizer = load_tokenizer()
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
    model_config.vocab_size = len(tokenizer)
    model_config.pad_token_id = tokenizer.pad_token_id

    model = LLaMAModel(model_config)
    model.to(training_config['device'])
    
    print(f"Using device: {training_config['device']}")
    
    optimizer = optim.AdamW(model.parameters(), lr=training_config['learning_rate'])
    total_steps = training_config['num_epochs'] * len(train_dataloader)
    print(f'Total steps: {total_steps}')
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training_config['warmup_steps'],
        num_training_steps=total_steps
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    train_model(
        model=model,
        train_loader=train_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        training_config=training_config,
        device=training_config['device']
    )
    
    model.save_pretrained(training_config['save_dir'])
    tokenizer.save_pretrained(training_config['save_dir'])
    print(f"Training completed. Model saved to '{training_config['save_dir']}'")


if __name__ == "__main__":
    main() 