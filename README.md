# LLaMA Implementation

This project represents an implementation of the LLaMA language model for an HSE NLP course. 

## Project Structure

```
llama/
├── src/                    # Core project code
│   ├── model/              # Model architecture module
│   │   ├── __init__.py
│   │   ├── llama_model.py  # Main LLaMA model
│   │   └── model_blocks.py # Model components (attention blocks, normalization, etc.)
│   ├── data/               # Data processing module
│   │   ├── __init__.py
│   │   └── data_processing.py # Data loading and preparation
│   ├── configs/            # Configurations
│   │   ├── __init__.py
│   │   ├── model_config.py # Model configuration
│   │   └── training_config.py # Training configuration
│   ├── training/           # Training module
│   │   ├── __init__.py
│   │   └── trainer.py      # Training functions
│   ├── inference/          # Text generation module
│   │   ├── __init__.py
│   │   └── generation.py   # Text generation functions
│   └── __init__.py         # Main initialization file
├── scripts/                # Launch scripts
│   ├── train.py            # Training script
│   └── inference.py        # Text generation script
├── checkpoints/            # Directory for saving model weights (created automatically)
└── data/                   # Data directory (created automatically)
```

## Dataset

The model is trained on the **ashaba1in/small_openwebtext** dataset, which is a smaller version of the OpenWebText corpus. OpenWebText consists of web texts collected from URLs shared on Reddit with high engagement. This dataset is suitable for language modeling tasks and is used for training text generation models like this LLaMA implementation.

The dataset is automatically downloaded during the training process and preprocessed to create fixed-length sequences for efficient training.

## Usage

### Training the Model

To train the model, run:

```bash
python scripts/train.py
```

Training will save model weights to the `checkpoints/` directory.

### Text Generation

To generate text using the trained model, run:

```bash
python scripts/inference.py --prompt "Your text here" --model_path "checkpoints"
```

Additional parameters:
- `--max_length` - maximum length of generated text (default 50)
- `--temperature` - sampling temperature (default 1.0)
- `--top_k` - top-k sampling parameter (default 50)
- `--top_p` - nucleus sampling parameter (default 0.95)
- `--repetition_penalty` - penalty for repetition (default 1.2)

## Dependencies

- PyTorch
- Transformers (Hugging Face)
- Datasets (Hugging Face)
- Wandb (for training logging)

## Model Architecture

The implementation includes the main LLaMA components:

- Self-Attention module with Rotary Positional Encoding
- RMSNorm normalization
- SwiGLU activations in feed-forward layers
- Efficient KV-cache for text generation
