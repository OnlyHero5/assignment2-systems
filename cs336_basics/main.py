# from train_bpe import BPE

# BPE.train_bpe(input_path="../data/TinyStoriesV2-GPT4-train.txt", 
#               vocab_size=10000, 
#               special_tokens=["<|endoftext|>"])

from fast_bpe_train_hf import train_bpelevel_bpe

train_bpelevel_bpe(input_path="../data/owt_train.txt",
                   vocab_size=32000,
                   special_tokens=["<|endoftext|>"],
                   output_path="../data/OWT")

train_bpelevel_bpe(input_path="../data/TinyStoriesV2-GPT4-train.txt",
                   vocab_size=10000,
                   special_tokens=["<|endoftext|>"],
                   output_path="../data/TinyStoriesV2")
