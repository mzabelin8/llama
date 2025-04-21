import torch
import argparse
from transformers import AutoTokenizer

from src.configs import LLaMAConfig
from src.model import LLaMAModel
from src.inference import generate_text


def main():
    parser = argparse.ArgumentParser(description="Generate text with LLaMA model")
    parser.add_argument("--model_path", type=str, default="checkpoints", help="Path to the model checkpoint")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Text prompt to start generation")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p (nucleus) sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Penalty for repetition")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        config = LLaMAConfig.from_pretrained(args.model_path)
        model = LLaMAModel.from_pretrained(args.model_path, config=config)
        model.to(device)
        model.eval()
        
        print(f"Model loaded from {args.model_path}")
        print(f"Generating with prompt: '{args.prompt}'")
        
        # Generate text
        generated_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=device
        )
        
        print("\nGenerated text:")
        print(generated_text)
        
    except Exception as e:
        print(f"Error loading model or generating text: {e}")


if __name__ == "__main__":
    main() 