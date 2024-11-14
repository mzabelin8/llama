
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Performs Root Mean Square Layer Normalization.
    """
    def __init__(self, dim, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.norm(dim=-1, keepdim=True) * (x.size(-1) ** -0.5)
        return self.scale * x / (rms + self.eps)



class SwiGLU(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * F.silu(x2)
    


class FeedForwardSwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size * 2, bias=False)  # 2x for SwiGLU
        self.activation = SwiGLU()
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))
    


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, input_tokens):
        return self.token_embedding(input_tokens)



class RotaryPositionalEncoding(nn.Module):
    def __init__(self, dim, config):
        super().__init__()
        self.dim = dim
        self.max_seq_len = config.max_position_embeddings
        self.base = 10000  # You can set this in config if needed

        # Precompute the inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def get_embed(self, seq_len, device, scaling_factor=1.0):
        # Compute positions with scaling factor
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype) * scaling_factor
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        emb = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return emb  # Shape: [seq_len, dim]

    def apply_rotary_emb(self, x, scaling_factor=1.0):
        batch_size, num_heads, seq_len, head_dim = x.size()
        # Get rotary embeddings
        rotary_emb = self.get_embed(seq_len, x.device, scaling_factor=scaling_factor).view(1, 1, seq_len, -1)
        x1, x2 = x.chunk(2, dim=-1)
        rotary_emb_cos, rotary_emb_sin = rotary_emb.chunk(2, dim=-1)

        x_rotated = torch.cat([
            x1 * rotary_emb_cos - x2 * rotary_emb_sin,
            x1 * rotary_emb_sin + x2 * rotary_emb_cos
        ], dim=-1)
        return x_rotated




class MultiQuerySelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim ** -0.5

        # Linear layers for Q, K, V
        self.query = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.key = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.out = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        # Rotary Positional Encoding
        self.rotary_pos_enc = RotaryPositionalEncoding(self.head_dim, config)

    def forward(self, x, kv_cache=None, scaling_factor=1.0):
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
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # Shape: [batch_size, num_heads, seq_len_q, seq_len_k]

        # Create causal mask
        causal_mask = torch.tril(torch.ones((seq_len_q, seq_len_k), device=Q.device)).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, seq_len_q, seq_len_k]

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
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = MultiQuerySelfAttention(config)
        self.norm2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.feed_forward = FeedForwardSwiGLU(config)

    def forward(self, x, kv_cache=None, scaling_factor=1.0):
        # Self-attention with residual connection
        attn_output, kv_cache = self.attention(self.norm1(x), kv_cache=kv_cache, scaling_factor=scaling_factor)
        x = x + attn_output

        # Feed-forward network with residual connection
        ff_output = self.feed_forward(self.norm2(x))
        x = x + ff_output

        return x, kv_cache