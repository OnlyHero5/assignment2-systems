from __future__ import annotations
import os
import argparse
import numpy as np
from pathlib import Path
from typing import Iterable, Dict, Any
from datasets import Dataset, load_dataset
from tokenizers import Tokenizer as HFTokenizer

import pyarrow as pa
import pyarrow.compute as pc

def prepare_dataset_with_hf(
        train_input : str | Path = None,
        valid_input : str | Path = None,
        input_path: str | Path = None,
        vocab_path: str | Path = None,
        eot_token: str = "<|endoftext|>",
        train_split: float = 0.9,
        out_path: str | Path = "../data",
        dtype: str = "np.uint16",
        num_proc: int = None
) -> None:
    """
    使用Hugging Face Datasets库高效处理数据集
    
    Args:
        input_path: 输入文件或目录路径
        tokenizer: 已训练的tokenizer
        eot_token: 文档结束标记
        train_split: 训练集比例
        output_dir: 输出目录
        dtype: 数据类型（uint16或int32）
        num_proc: 并行进程数（None表示自动）
    """
    print("="*60)
    print("使用huggingface datasets快速处理")
    print("="*60)

    #=============
    #1. 加载数据集
    #=============
    def load_text_dataset(path: str | Path) -> Dataset:
        path = Path(path)
        if path.is_dir():
            dataset = load_dataset(
                "text",
                data_files=str(path/"*.txt"),
                split="train",
                cache_dir="./cache",
            )
        else:
            dataset = load_dataset(
                "text",
                data_files=str(path),
                split="train",
                cache_dir="./cache",
            )
        return dataset

    #=============
    #2. tokenizer预处理
    #=============
    if vocab_path.endswith(".json") and "tokenizer" in str(vocab_path):
        tokenizer = HFTokenizer.from_file(str(vocab_path))
    else:
        from tokenizers import models, Tokenizer as HFTok
        tokenizer = HFTok(models.BPE.from_file(
            vocab=str(vocab_path),
            merges=str(vocab_path).replace("vocab.json", "merges.json")
        ))


    def tokenize_function(examples: Dict[str, list])-> Dict[str, list]:
        all_ids = []
        for text in examples["text"]:
            encoding = tokenizer.encode(text + eot_token)
            all_ids.append(encoding.ids)
        return {"input_ids": all_ids}
    
    #============
    #3. 展平数据集
    #============
    def flatten_function(examples: Dict[str, list]) -> Dict[str, list]:
        lengths = [len(x) for x in examples["input_ids"]]
        total = sum(lengths)

        buf = np.empty(total, dtype=np.int64)
        pos = 0
        for seq in examples["input_ids"]:
            buf[pos:pos+len(seq)] = seq
            pos += len(seq)
        return {"ids": [buf]}

    #=============
    #处理单个数据集
    #=============
    def process_dataset(dataset: Dataset, desc: str) -> np.ndarray:
        print(f"\n Tokenization ({desc}) ...")
        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=1000,
            num_proc=num_proc,
            remove_columns=dataset.column_names,
            desc=f"Tokenizing {desc}",
            cache_file_name="./cache/tokenized_train.arrow"
        )

        print(f"\n Flattening ({desc})...")
        flat = tokenized.map(
            flatten_function,
            batched=True,
            batch_size=10000,
            num_proc=num_proc,
            remove_columns=["input_ids"],
            desc=f"Flattening {desc} (chunked)",
            cache_file_name="./cache/flat_train.arrow"
        )

        dtype_map = {
            "np.uint16": np.uint16,
            "np.int32": np.int32,
            "uint16": np.uint16,
            "int32": np.int32,
        }
        target_dtype = dtype_map[dtype]

        print(f"\n 拼接 {desc} tokens...")

        arrow_table = flat.data.table
        ids_column = arrow_table.column("ids")

        all_chunks = []
        for chunk in ids_column.chunks:
            flattened_chunk = chunk.flatten()
            all_chunks.append(flattened_chunk)
        
        concatenated = pa.concat_arrays(all_chunks)

        if target_dtype == np.uint16:
            concatenated = pc.cast(concatenated, pa.uint16())
        elif target_dtype == np.int32:
            concatenated = pc.cast(concatenated, pa.int32())
        
        all_ids = concatenated.to_numpy()

        return all_ids

    #=============
    #4. 处理训练集和验证集
    #=============
    os.makedirs(out_path, exist_ok=True)
    train_path = os.path.join(out_path, "train.npy")
    valid_path = os.path.join(out_path, "valid.npy")

    if train_input and valid_input:
        print("\n加载训练集和验证集...")

        print("\n[1/2] 加载训练集...")
        print(f" 加载: {train_input}")
        train_dataset = load_text_dataset(train_input)
        print(f"  ✓ 加载了 {len(train_dataset)} 个文档")

        train_ids = process_dataset(train_dataset, desc="训练集")
        print(f"  ✓ 训练集token数: {len(train_ids):,}")

        print("\n[2/2] 加载验证集...")
        print(f" 加载: {valid_input}")
        valid_dataset = load_text_dataset(valid_input)
        print(f"  ✓ 加载了 {len(valid_dataset)} 个文档")

        valid_ids = process_dataset(valid_dataset, desc="验证集")
        print(f"  ✓ 验证集token数: {len(valid_ids):,}")
    
    elif input_path:
        print("\n✓ 模式B：单一输入，按比例划分")
        
        print("\n[1/1] 处理数据...")
        print(f"  加载: {input_path}")
        dataset = load_text_dataset(input_path)
        print(f"  ✓ 加载了 {len(dataset)} 个文档")
        
        all_ids = process_dataset(dataset, "all")
        total_tokens = len(all_ids)
        print(f"  ✓ 总token数: {total_tokens:,}")
        
        # 划分训练集和验证集
        print(f"\n  按比例划分: {train_split*100:.0f}% train / {(1-train_split)*100:.0f}% valid")
        split_idx = int(total_tokens * train_split)
        train_ids = all_ids[:split_idx]
        valid_ids = all_ids[split_idx:]
    
    else:
        raise ValueError("必须指定 (--train_input 和 --valid_input) 或 --input")


    #=================
    #5. 保存到文件
    #=================
    print("\n✓ 保存到文件...")
    print(f"  ✓ 训练集: {train_path}")
    print(f"  ✓ 验证集: {valid_path}")
    
    np.save(train_path, train_ids)
    np.save(valid_path, valid_ids)


def main():
    parser = argparse.ArgumentParser(
        description="使用Hugging Face Datasets库高效处理数据集"
    )

    # 输入配置
    parser.add_argument(
        "--train_input",
        type=str,
        help="训练集输入文件或目录路径",
        default=None,
    )
    parser.add_argument(
        "--valid_input",
        type=str,
        help="验证集输入文件或目录路径",
        default=None,
    )
    parser.add_argument(
        "--input",
        type=str,
        help="输入文件或目录路径",
        default=None,
    )
    parser.add_argument(
        "--split",
        type=float,
        help="训练集比例",
        default=0.9,
    )

    #输出配置
    parser.add_argument(
        "--out_dir",
        type=str,
        help="输出目录",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="uint16",
        choices=["uint16", "int32"],
        help="数据类型（uint16或int32）",
    )

    # Tokenizer配置
    parser.add_argument(
        "--vocab",
        type=str,
        help="Vocab文件路径",
        default="../data/vocab.json",
    )

    parser.add_argument(
        "--eot_token",
        type=str,
        help="文档结束标记",
        default="<|endoftext|>",
    )

    # 性能配置
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="并行进程数（None表示自动）",
    )

    args = parser.parse_args()


    # 处理数据集
    prepare_dataset_with_hf(
        train_input=args.train_input,
        valid_input=args.valid_input,
        input_path=args.input,
        vocab_path=args.vocab,
        eot_token=args.eot_token,
        train_split=args.split,
        out_path=args.out_dir,
        dtype=args.dtype,
        num_proc=args.num_workers,
    )

if __name__ == "__main__":
    main()