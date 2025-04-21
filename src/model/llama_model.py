import torch.nn as nn

from transformers import PreTrainedModel
from src.model.model_blocks import LLaMABlock, RMSNorm
from src.configs.model_config import LLaMAConfig


class LLaMAModel(PreTrainedModel):
    """
    LLaMA (Large Language Model Meta AI) implementation.
    
    This model is a PyTorch implementation of the LLaMA architecture,
    integrated with the Hugging Face Transformers library. It consists of
    a token embedding layer, a stack of transformer blocks, and a final
    normalization layer followed by an output projection.
    """
    config_class = LLaMAConfig

    def __init__(self, config):
        """
        Initialize the LLaMA model.
        
        Args:
            config: Model configuration object containing parameters
        """
        super().__init__(config)
        self.config = config

        # Token embedding layer
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

        # Stack of transformer blocks
        self.layers = nn.ModuleList([LLaMABlock(config) for _ in range(config.num_hidden_layers)])

        # Final layer normalization
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Output projection to vocabulary
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights
        self.post_init()

    def forward(self, input_tokens, kv_cache=None, scaling_factor=1.0):
        """
        Forward pass through the model.
        
        Args:
            input_tokens: Tensor of token IDs [batch_size, seq_len]
            kv_cache: Optional key-value cache for efficient inference
            scaling_factor: Position scaling factor for RoPE
            
        Returns:
            logits: Output logits tensor [batch_size, seq_len, vocab_size]
            kv_cache: Updated key-value cache for next forward pass
        """
        # Convert token IDs to embeddings
        x = self.embeddings(input_tokens)
        
        # Process through transformer blocks
        for layer in self.layers:
            x, kv_cache = layer(x, kv_cache=kv_cache, scaling_factor=scaling_factor)
        
        # Apply final normalization
        x = self.norm(x)
        
        # Project to vocabulary
        logits = self.output(x)
        
        return logits, kv_cache 