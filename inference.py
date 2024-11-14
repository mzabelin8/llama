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
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    kv_cache = None

    model.eval()

    with torch.no_grad():
        for _ in range(max_length):
            outputs, kv_cache = model(input_ids, kv_cache=kv_cache)
            logits = outputs[:, -1, :]  

            logits = logits / temperature

            if repetition_penalty != 1.0:
                for i in set(input_ids.view(-1).tolist()):
                    logits[0, i] /= repetition_penalty

            filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

            probabilities = torch.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text


def top_k_top_p_filtering(
    logits, top_k=0, top_p=0.0, filter_value=-float('Inf')
):
    top_k = min(top_k, logits.size(-1))  

    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, filter_value)

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
            ..., :-1
        ].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, filter_value)
    return logits