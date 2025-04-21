from transformers import PretrainedConfig


class LLaMAConfig(PretrainedConfig):
    """
    Configuration class for LLaMA model.
    
    This class stores the configuration parameters required for initializing the LLaMA model.
    It inherits from PretrainedConfig, making it compatible with the Hugging Face Transformers library.
    """
    model_type = "llama"

    def __init__(
        self,
        vocab_size=32000,        # Size of the vocabulary
        hidden_size=256,         # Dimension of the hidden layers
        num_hidden_layers=6,     # Number of transformer blocks
        num_attention_heads=8,   # Number of attention heads
        intermediate_size=1024,  # Dimension of the feed-forward layer
        rms_norm_eps=1e-6,       # Epsilon for RMSNorm
        max_position_embeddings=2048,  # Maximum sequence length
        initializer_range=0.02,  # Standard deviation for initializing weights
        pad_token_id=0,          # Token ID for padding
        **kwargs
    ):
        """
        Initialize a LLaMAConfig object.
        
        Args:
            vocab_size: Size of the vocabulary
            hidden_size: Dimension of the hidden layers
            num_hidden_layers: Number of transformer blocks
            num_attention_heads: Number of attention heads
            intermediate_size: Dimension of the feed-forward layer
            rms_norm_eps: Epsilon for RMSNorm
            max_position_embeddings: Maximum sequence length
            initializer_range: Standard deviation for initializing weights
            pad_token_id: Token ID for padding
            **kwargs: Additional keyword arguments passed to parent class
        """
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range


def get_default_model_config():
    """
    Create a default LLaMAConfig object with recommended settings.
    
    Returns:
        LLaMAConfig: Default configuration object
    """
    return LLaMAConfig(
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