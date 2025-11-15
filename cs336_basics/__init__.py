import importlib.metadata
from . import nn_utils
from . import data
from . import tokenizer
from .train_bpe import BPE
from .model import Linear, Embedding, RMSNorm, silu, RoPE, scaled_dot_product_attention, MultiHeadAttention, SwiGLU, TransformerBlock, TransformerLM
from .optimizer import AdamW, get_lr_cosine_schedule
from .serialization import save_checkpoint, load_checkpoint

try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    __version__ = "dev"  # 或者使用 "unknown" 或其他默认值
