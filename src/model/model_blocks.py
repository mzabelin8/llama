import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm) implementation.
    RMSNorm is a simplification of Layer Normalization that normalizes by the root
    mean square of the inputs, without centering by the mean.
    """
    def __init__(self, dim, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Calculate RMS
        rms = x.norm(dim=-1, keepdim=True) * (x.size(-1) ** -0.5)
        # Normalize and scale
        return self.scale * x / (rms + self.eps)



class SwiGLU(nn.Module):
    """
    SwiGLU activation function: x1 * SiLU(x2)
    Used as the non-linearity in the feed-forward layer.
    """
    def forward(self, x):
        # Split tensor in half along last dimension
        x1, x2 = x.chunk(2, dim=-1)
        # Apply SwiGLU: x1 * sigmoid(x2) * x2
        return x1 * F.silu(x2)
    


class FeedForwardSwiGLU(nn.Module):
    """
    Feed-forward layer with SwiGLU activation for LLaMA model.
    First projects hidden states to a higher dimension, applies SwiGLU,
    then projects back to the hidden size.
    """
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size * 2, bias=False)
        self.activation = SwiGLU()
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        # Project up, apply activation, project down
        return self.fc2(self.activation(self.fc1(x)))
    


class EmbeddingLayer(nn.Module):
    """
    Token embedding layer for the LLaMA model.
    Maps token IDs to embeddings.
    """
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, input_tokens):
        return self.token_embedding(input_tokens)



class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Positional Encoding (RoPE).
    Applies rotation to the key and query tensors based on their positions.
    This allows the model to consider token positions without explicit position embeddings.
    """
    def __init__(self, dim, config):
        super().__init__()
        self.dim = dim
        self.max_seq_len = config.max_position_embeddings
        self.base = 10000

        # Precompute the inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def get_embed(self, seq_len, device, scaling_factor=1.0):
        """
        Compute positional embeddings for a sequence.
        
        Args:
            seq_len: Length of the sequence
            device: Device for the tensors
            scaling_factor: Factor to scale positions for extrapolation
            
        Returns:
            Positional embeddings for the sequence
        """
        # Compute positions with scaling factor
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype) * scaling_factor
        # Compute sinusoidal encodings
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        # Return concatenated sin and cos embeddings
        emb = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return emb

    def apply_rotary_emb(self, x, scaling_factor=1.0):
        """
        Apply rotary embeddings to input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, num_heads, seq_len, head_dim]
            scaling_factor: Factor to scale positions for extrapolation
            
        Returns:
            Tensor with rotary embeddings applied
        """
        batch_size, num_heads, seq_len, head_dim = x.size()
        # Get rotary embeddings
        rotary_emb = self.get_embed(seq_len, x.device, scaling_factor=scaling_factor).view(1, 1, seq_len, -1)
        # Split input and embeddings
        x1, x2 = x.chunk(2, dim=-1)
        rotary_emb_cos, rotary_emb_sin = rotary_emb.chunk(2, dim=-1)

        # Apply rotation using the rotation matrix:
        # [cos, -sin]
        # [sin,  cos]
        x_rotated = torch.cat([
            x1 * rotary_emb_cos - x2 * rotary_emb_sin,
            x1 * rotary_emb_sin + x2 * rotary_emb_cos
        ], dim=-1)
        return x_rotated




class MultiQuerySelfAttention(nn.Module):
    """
    Multi-Query Self-Attention module with Rotary Positional Encoding.
    Computes self-attention with a causal mask and uses a key-value cache
    for efficient autoregressive generation.
    """
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim ** -0.5

        # Linear projections for Q, K, V
        self.query = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.key = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.out = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        # Rotary Positional Encoding
        self.rotary_pos_enc = RotaryPositionalEncoding(self.head_dim, config)

    def forward(self, x, kv_cache=None, scaling_factor=1.0):
        """
        Forward pass through the attention module.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            kv_cache: Optional key-value cache for efficient inference
            scaling_factor: Position scaling factor for RoPE
            
        Returns:
            Output tensor and updated key-value cache
        """
        batch_size, seq_len, _ = x.size()

        # Compute Q, K, V
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply rotary positional encoding
        Q = self.rotary_pos_enc.apply_rotary_emb(Q, scaling_factor=scaling_factor)
        K = self.rotary_pos_enc.apply_rotary_emb(K, scaling_factor=scaling_factor)


        # Use KV cache if provided (for autoregressive generation)
        if kv_cache is not None:
            K = torch.cat([kv_cache['K'], K], dim=2)
            V = torch.cat([kv_cache['V'], V], dim=2)
            kv_cache['K'], kv_cache['V'] = K, V
        else:
            kv_cache = {'K': K, 'V': V}

        seq_len_q = Q.size(2)
        seq_len_k = K.size(2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Create causal mask
        causal_mask = torch.tril(torch.ones((seq_len_q, seq_len_k), device=Q.device)).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        # Apply mask
        attn_scores = attn_scores.masked_fill(~causal_mask, float('-inf'))

        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Attention output
        output = torch.matmul(attn_weights, V)

        # Concatenate heads and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, -1)
        return self.out(output), kv_cache



class LLaMABlock(nn.Module):
    """
    LLaMA Transformer block with pre-layer normalization.
    Consists of a self-attention layer followed by a feed-forward layer,
    with residual connections around each.
    """
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = MultiQuerySelfAttention(config)
        self.norm2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.feed_forward = FeedForwardSwiGLU(config)

    def forward(self, x, kv_cache=None, scaling_factor=1.0):
        """
        Forward pass through the transformer block.
        
        Args:
            x: Input tensor
            kv_cache: Optional key-value cache
            scaling_factor: Position scaling factor
            
        Returns:
            Output tensor and updated key-value cache
        """
        # Self-attention with residual connection
        attn_output, kv_cache = self.attention(self.norm1(x), kv_cache=kv_cache, scaling_factor=scaling_factor)
        x = x + attn_output

        # Feed-forward network with residual connection
        ff_output = self.feed_forward(self.norm2(x))
        x = x + ff_output

        return x, kv_cache 