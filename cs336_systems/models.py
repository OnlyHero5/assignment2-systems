import torch
from cs336_basics.model import TransformerLM
from typing import Dict, Any

MODEL_CONFIGS = {
    "small": {
        "d_model": 768,
        "num_layers": 12,
        "num_heads": 12,
        "d_ff": 3072,  # 4 * d_model
    },
    "medium": {
        "d_model": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "d_ff": 4096,  # 4 * d_model
    },
    "large": {
        "d_model": 1280,
        "num_layers": 32,
        "num_heads": 20,
        "d_ff": 5120,  # 4 * d_model
    },
    "xl": {
        "d_model": 1600,
        "num_layers": 40,
        "num_heads": 25,
        "d_ff": 6400,  # 4 * d_model
    },
    "2.7B": {
        "d_model": 2560,
        "num_layers": 32,
        "num_heads": 32,
        "d_ff": 10240,  # 4 * d_model
    }
}

def get_basics_transformer(
        size: str,
        context_length: int,
        vocab_size: int,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda"
) -> TransformerLM:
    """
    Create a Transformer model with predefined configuration.
    
    Args:
        size: Model size. One of ["small", "medium", "large", "xl", "2.7B"]
        context_len: Maximum sequence length
        vocab_size: Vocabulary size (default: 10000)
        dtype: Data type for model parameters (default: torch.float32)
        device: Device to place the model on (default: "cuda")
    
    Returns:
        TransformerLM model instance
    
    Example:
        >>> model = get_basics_transformer("small", 512, 10000, torch.float32, "cuda")
        >>> print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    """
    if size not in MODEL_CONFIGS:
        raise ValueError(
            f"Invalid size {size}."
            f"Choose from {list(MODEL_CONFIGS.keys())}."
        )
    config = MODEL_CONFIGS[size]

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
    )
    model.to(dtype=dtype, device=device)
    return model

def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count the number of parameters in the model.
    
    Args:
        model: TransformerLM model instance
    
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": total_params - trainable_params
    }


if __name__ == "__main__":
    print("Testing model configurations...\n")
    for size in MODEL_CONFIGS.keys():
        model = get_basics_transformer(
            size=size,
            context_length=512,
            vocab_size=10000,
            dtype=torch.float32,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        params = count_parameters(model)
        param_str = f"{params['total']:,}"
        print(f"{size:8s}: {param_str:>12} parameters")
    
    print("\n✓ All model configurations created successfully!")