from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tokenizers.processors import TemplateProcessing
from pathlib import Path
import json



def train_bpelevel_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str] | None = None,
    min_frequency: int = 2,
    output_path: str | Path  = "../data",
) :
    
    save_dir = Path(output_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )
    tokenizer.train([str(input_path)], trainer=trainer)
    tokenizer.save(str(save_dir / "tokenizer.json"))

    with open(save_dir / "tokenizer.json", "r", encoding="utf-8") as f:
        tokenizer_data = json.load(f)
    
    vocab = tokenizer_data["model"]["vocab"]
    merges = tokenizer_data["model"]["merges"]

    with open(save_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)

    with open(save_dir / "merges.json", "w", encoding="utf-8") as f:
        json.dump(merges, f, ensure_ascii=False, indent=4)
 