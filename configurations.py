
from transformers import PretrainedConfig



class LLaMAConfig(PretrainedConfig):
    model_type = "llama"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=256,            
        num_hidden_layers=6,        
        num_attention_heads=8,      
        intermediate_size=1024,     
        rms_norm_eps=1e-6,
        max_position_embeddings=2048,
        initializer_range=0.02,
        pad_token_id=0,             
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range

