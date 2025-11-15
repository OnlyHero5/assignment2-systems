"""
BPE (Byte-Pair Encoding) Training Implementation

This module implements the training algorithm for Byte-Pair Encoding tokenizers,
following the approach used in GPT-2.

Algorithm Overview:
    1. Pre-tokenize corpus using regex
    2. Initialize vocab with 256 bytes + special tokens
    3. Iteratively merge most frequent adjacent pairs
    4. Record merge history for later use

References:
    - GPT-2 Paper: https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
    - Original BPE Paper: https://arxiv.org/abs/1508.07909
    - SentencePiece: https://github.com/google/sentencepiece

BPE重构    
Author: PSX
Date: 2025-10-31

"""
import os
import regex
import pickle
from collections import defaultdict
from typing import Dict, List, Set, Tuple
from pathlib import Path

class BPE:

    # 辅助函数: 合并token序列中的指定对
    def merge_token_sequence(
            token_sequence: Tuple[bytes, ...],
            merge_pair: Tuple[bytes, bytes],
            new_token: bytes
            ) -> Tuple[bytes, ...]:
        
        new_seq = []
        i = 0
        while i < len(token_sequence):
            if i < len(token_sequence) - 1 and token_sequence[i:i+2] == merge_pair:
                new_seq.append(new_token)
                i += 2  # 跳过下一个token
            else:
                new_seq.append(token_sequence[i])
                i += 1
        return tuple(new_seq)
    
    # 辅助函数: 词典初始化
    def _initialize_vocab(
            special_tokens: List[str],
            vocab_size: int,
    ) -> Tuple[Dict[bytes, int], int]:
        
        vocab: Dict[int, bytes] = {
            i: bytes([i]) for i in range(256)
        }
        next_id = 256

        existing_bytes: Set[bytes] = set(vocab.values())

        for token_str in special_tokens:
            if len(vocab) >= vocab_size:
                break
            
            token_bytes = token_str.encode('utf-8')

            if token_bytes not in existing_bytes:
                vocab[next_id] = token_bytes
                existing_bytes.add(token_bytes)
                next_id += 1
        
        return vocab, next_id
    

    # 将文本按特殊token和GPT2分词规则分割
    PRE_TOKENIZE_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    def _load_and_pretokenize(
            input_patn: str | os.PathLike,
            special_tokens: List[str],
    ) -> Dict[Tuple[bytes, ...], int]:
        
        try:
            with open(input_patn, 'r', encoding='utf-8') as f:
                text = f.read()
        except FileNotFoundError:
            text = ""
       
        if special_tokens:
            special_pattern = '|'.join(
                regex.escape(token) for token in special_tokens
            )
            # 按特殊token分割文本，且不捕获特殊token
            chunks = regex.split(special_pattern, text)
        else:
            chunks = [text]
        
        token_frequencies: Dict[Tuple[bytes, ...], int] = defaultdict(int)

        for chunk in chunks:
            # 进一步按GPT2规则分割
            words = regex.findall(BPE.PRE_TOKENIZE_PATTERN, chunk)

            for word in words:
                word_bytes = word.encode('utf-8')
                byte_sequence = tuple([bytes([b]) for b in word_bytes])
                token_frequencies[byte_sequence] += 1

        return token_frequencies
    
    # 统计token对频率
    def _count_token_pairs(
            token_frequencies: Dict[Tuple[bytes,...], int]
            ) -> Dict[Tuple[bytes, bytes], int]:
        pair_freq: Dict[Tuple[bytes, bytes], int] = defaultdict(int)

        for token_seq, seq_freq in token_frequencies.items():
            for i in range(len(token_seq) - 1):
                pair = (token_seq[i], token_seq[i+1])
                pair_freq[pair] += seq_freq
        
        return pair_freq
    
    # 查找出现频率最高的token对
    def _find_most_frequent_pair(
            pair_freq: Dict[Tuple[bytes, bytes], int]
    ) -> Tuple[bytes, bytes] | None:
        
        if not pair_freq:
            return None
        
        max_freq = max(pair_freq.values())
        candidates = [pair for pair, freq in pair_freq.items() if freq == max_freq]

        return max(candidates)  # 返回字典序最大的对
    
    # 执行合并操作
    def _perform_merge(
            token_freq: Dict[Tuple[bytes,...], int],
            pair_freq: Dict[Tuple[bytes, bytes], int],
            best_pair: Tuple[bytes, bytes],
            new_token: bytes
    ) -> None:
        
        affected_seqs = []

        # 查找所有包含best_pair的token序列
        for token_seq, freq in token_freq.items():
            has_pair = any(
                token_seq[i:i+2] == best_pair for i in range(len(token_seq) - 1)
            )
            if has_pair:
                affected_seqs.append((token_seq, freq))
        
        for old_seq, seq_freq in affected_seqs:
            for i in range(len(old_seq) - 1):
                old_pair = (old_seq[i], old_seq[i+1])
                pair_freq[old_pair] -= seq_freq
                if pair_freq[old_pair] <= 0:
                    del pair_freq[old_pair]
            
            # 生成新序列
            new_seq = BPE.merge_token_sequence(old_seq, best_pair, new_token)
            for i in range(len(new_seq) - 1):
                new_pair = (new_seq[i], new_seq[i+1])
                pair_freq[new_pair] += seq_freq
            
            # 更新token_freq: 移除旧序列，添加新序列
            del token_freq[old_seq]
            token_freq[new_seq] = seq_freq

    # 训练BPE
    @staticmethod
    def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    save_dir: str | os.PathLike = "../data", 
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        
        # ============================================================
        # 步骤 0: 参数验证
        # ============================================================
        if not isinstance(vocab_size, int) or vocab_size <= 0:
            raise ValueError("vocab_size 必须是正整数")
        if not isinstance(special_tokens, list):
            raise ValueError("special_tokens 必须是字符串列表")
        if not all(isinstance(token, str) for token in special_tokens):
            raise ValueError("special_tokens 中的所有元素必须是字符串")
        
        
        print("=" * 60)
        print("开始训练 BPE Tokenizer")
        print("=" * 60)
        print(f"目标词汇表大小: {vocab_size}")
        print(f"特殊 tokens: {special_tokens}")


        # ============================================================
        # 步骤 1: 初始化词汇表
        # ============================================================
        # (vocab, next_token_id) : Tuple[Dict[bytes, int], int]
        vocab, next_token_id = BPE._initialize_vocab(special_tokens, vocab_size)

        print(f"\n✓ 步骤 1: 初始化词汇表")
        print(f"  - 基础字节: 256")
        print(f"  - 特殊 tokens: {len(vocab) - 256}")
        print(f"  - 当前词汇表大小: {len(vocab)}")

        # ============================================================
        # 步骤 2: 加载和预分词
        # ============================================================
        token_frequencies :Dict[Tuple[bytes, ...], int] = BPE._load_and_pretokenize(input_path, special_tokens)

        print(f"\n✓ 步骤 2: 加载和预分词")
        print(f"  - 唯一 token 序列数: {len(token_frequencies)}")
        total_tokens = sum(len(seq) * freq for seq, freq in token_frequencies.items())
        print(f"  - 总 token 数: {total_tokens}")

        # ============================================================
        # 步骤 3: 统计初始 token 对频率
        # ============================================================
        pair_frequencies: Dict[Tuple[bytes, bytes], int] = BPE._count_token_pairs(token_frequencies)
        
        print(f"\n✓ 步骤 3: 统计 token 对")
        print(f"  - 唯一 token 对数: {len(pair_frequencies)}")

        # ============================================================
        # 步骤 4: 迭代合并
        # ============================================================
        megres: List[Tuple[bytes, bytes]] = []
        iteration = 0

        print("\n✓ 步骤 4: 迭代合并")
        print(f"需要添加 {vocab_size - len(vocab)} 个新 tokens")

        while len(vocab) < vocab_size:
            iteration += 1

            # 查找出现频率最高的 token 对
            best_pair: Tuple[bytes, bytes] = BPE._find_most_frequent_pair(pair_frequencies)
            if best_pair is None:
                print("没有更多可合并的 token 对，提前结束")
                break
            
            # 创建新token
            new_token: bytes = best_pair[0] + best_pair[1]
            vocab[next_token_id] = new_token
            megres.append(best_pair)
            next_token_id += 1

            if iteration <= 5 or iteration % 100 == 0 or len(vocab) == vocab_size:
                freq = pair_frequencies[best_pair]
                try:
                    token_str = new_token.decode('utf-8', errors='replace')
                except:
                    token_str = str(new_token)
                print(f"  - 迭代 {iteration}: 合并 {best_pair[0]} + {best_pair[1]} = '{token_str}' (频率: {freq})"
                      f" 词汇表: {len(vocab):5d}/{vocab_size}")
            
            # 执行合并操作
            BPE._perform_merge(token_frequencies, pair_frequencies, best_pair, new_token)

        # =========================
        # 保存BPE结果进入本地磁盘
        # =========================
        data_dir = Path(save_dir)
        base_name = Path(input_path).stem
        data_dir.mkdir(parents=True, exist_ok=True)
        vocab_path = data_dir / f"{base_name}_vocab.pkl"
        merges_path = data_dir / f"{base_name}_merges.pkl"
        with open(vocab_path, "wb") as f:
            pickle.dump(vocab, f)
        with open(merges_path, "wb") as f:
            pickle.dump(megres, f) 
        
        print("\n" + "=" * 60)
        print("✓ BPE 训练完成")
        print("=" * 60)
        print(f"  - 最终词汇表大小: {len(vocab)}")
        print(f"  - 合并操作数: {len(megres)}")
        print(f"  - 词汇表已保存至: {vocab_path}")
        print(f"  - 合并记录已保存至: {merges_path}")

        return vocab, megres
                
                
            
    