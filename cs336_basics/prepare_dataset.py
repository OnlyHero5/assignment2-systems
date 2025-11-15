from __future__ import annotations
import os
import argparse
import numpy as np
from tokenizer import Tokenizer
from pathlib import Path
import tqdm


def _iter_texts(p: Path):
    if p.is_dir():
        for fp in sorted(p.rglob("*.txt")):
            with open(fp, "r", encoding="utf-8") as f:
                yield f.read()
    else:
        with open(p, "r", encoding="utf-8") as f:
            yield f.read()

def _encode_corpus_to_1d_ids(corpus_path: str, tok: Tokenizer, eot: str) -> np.ndarray:
    """
    对语料进行编码，返回1维的id序列，每个id代表一个token。
    """
    ids = []
    for doc in _iter_texts(Path(corpus_path)):
        ids.extend(tok.encode(doc))
        ids.extend(tok.encode(eot))
    return np.asarray(ids, dtype=np.uint16)

def _save_npy(path: str, arr: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.save(path, arr)



def main():
    ap = argparse.ArgumentParser("tokenized dataset")
    #=====================================
    # 如果是多文件，则指定训练集和验证集
    ap.add_argument("--train_input", type=str, help="训练语料库")
    ap.add_argument("--valid_input", type=str, help="验证语料库")
    #=====================================
    # 如果是单文件，则划分
    ap.add_argument("--input", type=str,  help="input path")
    ap.add_argument("--split", type=float, default=0.9, help="训练集比例,默认0.9")
    #=====================================
    ap.add_argument("--vocab", type=str, required=True, help="vocab path")
    ap.add_argument("--merges", type=str, required=True, help="merges path")
    ap.add_argument("--outdir", type=str, default="../data", help="output path")
    ap.add_argument("--dtype", type=str, default="np.uint16", choices=["np.int32", "np.uint16"], help="dtype")
    ap.add_argument("--eot", type=str, default="<|endoftext|>", help="end of text token")
    args = ap.parse_args()

    # 加载tokenizer
    tok = Tokenizer.from_file(args.vocab, args.merges, special_tokens=[args.eot])

    dtype_map = {
        "np.int32": np.int32,
        "np.uint16": np.uint16,
    }
    dtype = dtype_map[args.dtype]

    os.makedirs(args.outdir, exist_ok=True)
    train_out = os.path.join(args.outdir, "train.npy")
    valid_out = os.path.join(args.outdir, "valid.npy")

    if args.train_input and args.valid_input:
        # 多文件
        print("✓ 模式 A：分别编码 train / val（不再二次划分）")
        train_ids = _encode_corpus_to_1d_ids(args.train_input, tok, args.eot)
        val_ids = _encode_corpus_to_1d_ids(args.valid_input, tok, args.eot)
        train_ids = train_ids.astype(dtype)
        val_ids = val_ids.astype(dtype)
        _save_npy(train_out, train_ids)
        _save_npy(valid_out, val_ids)
        print(f"      - train tokens: {len(train_ids):,} -> {train_out}")
        print(f"      - valid tokens: {len(val_ids):,} -> {valid_out}")
    elif args.input:
        print("✓ 模式 B：单一输入，按比例切分")
        all_ids = _encode_corpus_to_1d_ids(args.input, tok, args.eot).astype(dtype)
        n = len(all_ids)
        n_train = int(n * float(args.split))
        _save_npy(train_out, all_ids[:n_train])
        _save_npy(valid_out, all_ids[n_train:])
        print(f"      - all tokens: {n:,} -> {train_out} + {valid_out}")
        print(f"      - train tokens: {n_train:,} -> {train_out}")
        print(f"      - valid tokens: {n - n_train:,} -> {valid_out}")
    else:
        raise ValueError("必须指定 --train_input 和 --valid_input 或 --input")
    
    print("✓ 完成。后续请用 np.load(..., mmap_mode='r') 在训练脚本中按需加载。")

if __name__ == "__main__":
    main()


    