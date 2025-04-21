#!/usr/bin/env python3

"""
Main script for launching LLaMA model training or text generation.
For backward compatibility.
"""

import sys
import os

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "inference":
        # Run inference
        from scripts import inference
        sys.argv.pop(1)  # Remove 'inference' argument
        inference.main()
    else:
        # Run training
        from scripts import train
        train.main()
        
    print("\nTip: To control execution, use scripts directly:")
    print("  Train model: python scripts/train.py")
    print("  Generate text: python scripts/inference.py --prompt 'Your text here'")

