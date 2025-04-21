import torch


def get_default_training_config():
    """
    Create a default training configuration dictionary.
    
    This function returns a dictionary with recommended training hyperparameters,
    including batch size, number of epochs, learning rate, etc.
    
    Returns:
        dict: Dictionary containing training configuration parameters
    """
    return {
        'batch_size': 8,              # Batch size for training
        'num_epochs': 1,              # Number of training epochs
        'save_freq': 10000,           # Steps between checkpoints
        'log_freq': 100,              # Steps between logging
        'save_dir': 'checkpoints',    # Directory to save checkpoints
        'learning_rate': 1e-4,        # Initial learning rate
        'warmup_steps': 500,          # Learning rate warmup steps
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Training device
    } 