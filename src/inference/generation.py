import torch

def generate_text(
    model,
    tokenizer,
    prompt,
    max_length=50,
    temperature=1.0,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.0,
    device='cpu'
):
    """
    Generate text using the LLaMA model.
    
    Args:
        model: LLaMA model
        tokenizer: Tokenizer for encoding/decoding text
        prompt (str): Input prompt to start generation
        max_length (int): Maximum number of tokens to generate
        temperature (float): Sampling temperature (1.0 = no change, <1.0 = less random, >1.0 = more random)
        top_k (int): Number of highest probability tokens to keep for top-k sampling
        top_p (float): Cumulative probability threshold for nucleus sampling
        repetition_penalty (float): Penalty for repeating tokens (1.0 = no penalty)
        device (str): Device to run generation on ('cpu' or 'cuda')
        
    Returns:
        str: Generated text including the prompt
    """
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    kv_cache = None

    model.eval()

    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            outputs, kv_cache = model(input_ids, kv_cache=kv_cache)
            logits = outputs[:, -1, :]  

            # Apply temperature
            logits = logits / temperature

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in set(input_ids.view(-1).tolist()):
                    logits[0, i] /= repetition_penalty

            # Apply top-k and top-p filtering
            filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

            # Sample next token
            probabilities = torch.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)

            # Add next token to input sequence
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # Stop if end of sequence token is generated
            if next_token.item() == tokenizer.eos_token_id:
                break

    # Decode the generated sequence
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text


def top_k_top_p_filtering(
    logits, top_k=0, top_p=0.0, filter_value=-float('Inf')
):
    """
    Filter logits using top-k and/or top-p (nucleus) filtering.
    
    Args:
        logits: Logits distribution (batch_size, vocabulary_size)
        top_k (int): Keep only the top-k tokens with highest probability (top-k filtering)
        top_p (float): Keep the top tokens with cumulative probability >= top_p (nucleus filtering)
        filter_value (float): Value to assign to filtered tokens
        
    Returns:
        torch.Tensor: Filtered logits
    """
    top_k = min(top_k, logits.size(-1))  

    # Top-k filtering
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, filter_value)

    # Top-p (nucleus) filtering
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift indices to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
            ..., :-1
        ].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter sorted tensors back to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, filter_value)
    return logits 