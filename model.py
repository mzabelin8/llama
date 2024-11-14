import torch.nn as nn


from transformers import PreTrainedModel
from model_blocks import LLaMABlock, RMSNorm
from configurations import LLaMAConfig


class LLaMAModel(PreTrainedModel):
    config_class = LLaMAConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Embedding layer
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer blocks
        self.layers = nn.ModuleList([LLaMABlock(config) for _ in range(config.num_hidden_layers)])

        # Final RMSNorm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Output projection layer
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def forward(self, input_tokens, kv_cache=None, scaling_factor=1.0):
        x = self.embeddings(input_tokens)
        for layer in self.layers:
            x, kv_cache = layer(x, kv_cache=kv_cache, scaling_factor=scaling_factor)
        x = self.norm(x)
        logits = self.output(x)
        return logits, kv_cache
    
    