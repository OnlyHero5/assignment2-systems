
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import chain
from multiprocessing import Pool, Pipe, Process
import os
import pickle
from typing import Any, Callable, DefaultDict, Dict, Iterable, Iterator, List, Optional, Self, Set, Tuple, TypeVar, Union

import regex as re
from sortedcontainers import SortedSet
from tqdm import tqdm

from cs336_basics.pretokenization_example import find_chunk_boundaries

@dataclass
class BPEParams:
    vocab: Dict[int, bytes]
    merges: List[Tuple[bytes, bytes]]

@dataclass
class TokenEntry:
    token: int
    # these are intended to be stored in a list, with pointers to list indices rather than objects,
    # to allow for the potential of easier serialization across processes
    prev: Optional[int]
    next: Optional[int]

@dataclass
class TokenSplit:
    # represent these as arrays of primitives for much faster serialization during multiprocessing
    tokens: List[int] = field(default_factory=list)
    prev_ptrs: List[int] = field(default_factory=list)
    next_ptrs: List[int] = field(default_factory=list)
    byte_pairs: Dict[Tuple[int, int], Set[int]] = field(default_factory=dict)


DeltaSet = Dict[Tuple[int, int], int]

class TokenizerTrainingManager:
    def __init__(self):
        self.special_tokens: Set[str] = set()
        self.tokens: List[int] = []
        self.prev_ptrs: List[int] = []
        self.next_ptrs: List[int] = []
        self.pair_positions: DefaultDict[Tuple[int, int], Set[int]] = defaultdict(set)

    def set_special_tokens(self, tokens: Iterable[str]) -> None:
        self.special_tokens = set(tokens)

    def _presplit_tokens(self, text: str) -> None:
        TOKEN_REGEX = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for match in re.finditer(TOKEN_REGEX, text):
            s = match.group(0)
            if not s:
                continue

            s_bytes = bytes(s, encoding="utf-8")
            base = len(self.tokens)
            for i, b in enumerate(s_bytes):
                self.tokens.append(b)
                self.prev_ptrs.append(base+i-1 if i > 0 else -1)
                self.next_ptrs.append(base+i+1 if i < len(s_bytes) - 1 else -1)
                if i > 0:
                    byte_pair = (s_bytes[i-1], s_bytes[i])
                    self.pair_positions[byte_pair].add(base+i-1)

    def _split_text_into_docs(self, all_docs: str) -> List[str]:
        special_token_regex = re.compile("|".join(self.special_tokens))
        return re.split(special_token_regex, all_docs)

    def presplit_from_file(self, path: str | os.PathLike, start_index: int, end_index: int) -> DeltaSet:
        with open(path) as input_file:
            input_file.seek(start_index)
            input_data = input_file.read(end_index - start_index)
        
        for doc in self._split_text_into_docs(input_data):
            self._presplit_tokens(doc)

        return {pair : len(positions) for pair, positions in self.pair_positions.items()}

    def merge_tokens(self, old_pair: Tuple[int, int], new_token: int) -> DeltaSet:
        t1, t2 = old_pair

        pair_adds: DefaultDict[Tuple[int, int], Set[int]] = defaultdict(set)
        pair_removals: DefaultDict[Tuple[int, int], Set[int]] = defaultdict(set)

        # iterate in ascending order because some updates can end up removing later entries in the collection
        # (e.g. tokens X, X, X when we're about to merge X + X = XX)
        for index in sorted(self.pair_positions[old_pair]):
            pair_removals[old_pair].add(index)

            if self.tokens[index] == -1:
                # this means we already cleared out this entry from a previous merge in this generation
                continue

            assert self.tokens[index] == t1
            assert self.next_ptrs[index] > 0
            next_entry: int = self.next_ptrs[index]
            assert self.tokens[next_entry] == t2

            # merge tokens into first element of pair
            self.tokens[index] = new_token
            old_next_index: int = self.next_ptrs[index]
            self.next_ptrs[index] = self.next_ptrs[old_next_index]
            if self.next_ptrs[index] is not None:
                self.prev_ptrs[self.next_ptrs[index]] = index

            # tombstone second element of pair so that it isn't re-processed
            self.tokens[old_next_index] = -1

            # add/remove pair counts based on merge
            if self.prev_ptrs[index] >= 0:
                pair_adds[(self.tokens[self.prev_ptrs[index]], self.tokens[index])].add(self.prev_ptrs[index])
                pair_removals[(self.tokens[self.prev_ptrs[index]], t1)].add(self.prev_ptrs[index])
            if self.next_ptrs[index] >= 0:
                pair_removals[(t2, self.tokens[self.next_ptrs[index]])].add(old_next_index)
                pair_adds[(self.tokens[index], self.tokens[self.next_ptrs[index]])].add(index)

        pair_deltas: DefaultDict[Tuple[int, int], int] = defaultdict(int)
        for pair, entries in pair_adds.items():
            pair_deltas[pair] += len(entries)
            for added_entry in entries:
                self.pair_positions[pair].add(added_entry)

        for pair, entries in pair_removals.items():
            pair_deltas[pair] -= len(entries)
            for added_entry in entries:
                self.pair_positions[pair].remove(added_entry)
    
        return pair_deltas

def worker_main(shard: int, pipe):
    manager = TokenizerTrainingManager()

    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(os.getpid(), {shard})

    while True:
        fname, args = pipe.recv()
        try:
            if fname in ("stop", "exit"):
                pipe.close()
                break
            elif fname == "set_special_tokens":
                manager.set_special_tokens(args[0])
            elif fname == "load_from_file":
                deltas = manager.presplit_from_file(*args)
                pipe.send(deltas)
            elif fname == "merge_tokens":
                deltas = manager.merge_tokens(*args)
                pipe.send(deltas)
        except Exception as e:
            pipe.send(f"error: {str(e)}")


class TokenizerTrainer:
    def __init__(self, vocab_size: int, special_tokens: Iterable[str], debug: bool = False):
        self.vocab_size: int = vocab_size
        self.special_tokens: Set[str] = set(special_tokens)
        self._debug = debug

    def train_on_file(self, path: Union[str, os.PathLike], n_workers: Optional[int] = None) -> BPEParams:
        n_workers = n_workers or os.cpu_count()
        assert n_workers is not None

        with open(path, "rb") as input_file:
            chunk_boundaries = find_chunk_boundaries(input_file, n_workers, b"<|endoftext|>")

        chunks: List[Tuple[int, int]] = list(zip(chunk_boundaries[:-1], chunk_boundaries[1:]))
        assert len(chunks) <= n_workers
        n_workers = len(chunks)

        vocab: List[bytes] = [bytes([b]) for b in range(256)]
        token_lookup: Dict[bytes, int] = { token: i for i, token in enumerate(vocab) }

        def _add_token(token: bytes):
            assert token not in token_lookup
            token_lookup[token] = len(vocab)
            vocab.append(token)

        merges: List[Tuple[bytes, bytes]] = []
        byte_pair_counts: DefaultDict[Tuple[int, int], int] = defaultdict(int)

        def pair_sort_key(x: Tuple[int, Tuple[int, int]]) -> Tuple[int, bytes, bytes]:
            return (x[0], vocab[x[1][0]], vocab[x[1][1]])
        top_counts: SortedSet[Tuple[int, Tuple[int, int]]] = SortedSet(key=pair_sort_key) # type: ignore

        def _update_pair_count(pair: Tuple[int, int], delta: int) -> None:
            if byte_pair_counts[pair] > 0:
                top_counts.remove((byte_pair_counts[pair], pair))
            byte_pair_counts[pair] += delta
            if byte_pair_counts[pair] > 0:
                top_counts.add((byte_pair_counts[pair], pair))

        # TODO:
        # 1. move worker construction/init to dedicated functions and separate it from business logic
        # 2. switch to manager pattern so that worked nodes can run across multiple machines, allows for wider scale
        #   2a. service discovery for these nodes
        #   2b. recovery operations to catch up or replace dead nodes (i.e. reload base state and re-apply all previous merges)

        pipes = [Pipe() for _ in range(n_workers)]
        local_pipes = [pipe[0] for pipe in pipes]
        remote_pipes = [pipe[1] for pipe in pipes]
        workers = [Process(target=worker_main, args=(i, pipe)) for i, pipe in enumerate(remote_pipes)]
        
        try:
            # TODO: instead of using finally block to clean up, create context to automatically clean up based on scope
            for worker in workers:
                worker.start()

            for chunk, pipe in zip(chunks, local_pipes):
                pipe.send(("set_special_tokens", (self.special_tokens,)))
                pipe.send(("load_from_file", (path, chunk[0], chunk[1])))
            for pipe in local_pipes:
                deltas = pipe.recv()
                for pair, count in deltas.items():
                    _update_pair_count(pair, count)

            with tqdm(total=self.vocab_size, desc="merging tokens...") as pbar:
                while len(vocab) < self.vocab_size - len(self.special_tokens):
                    pbar.update(len(vocab) - pbar.n)
                    if len(top_counts) == 0:
                        print(f"exhausted all token pairs with vocab size of only {len(vocab)}")
                        break
                    
                    _, top_pair = next(reversed(top_counts))
                    t1, t2 = top_pair

                    # register new token
                    merges.append((vocab[t1], vocab[t2]))
                    new_token: bytes = vocab[t1] + vocab[t2]
                    _add_token(new_token)
                    new_token_id: int = token_lookup[new_token]

                    # merge in each shard and update counts accordingly
                    for pipe in local_pipes:
                        pipe.send(("merge_tokens", (top_pair, new_token_id)))
                    for pipe in local_pipes:
                        deltas = pipe.recv()
                        # XXX: create wrapper to check for errors anytime we receive call results
                        if isinstance(deltas, str):
                            raise Exception(deltas)
                        for pair, count in deltas.items():
                            _update_pair_count(pair, count)

                for special_token in self.special_tokens:
                    _add_token(bytes(special_token, encoding="utf-8"))
                    pbar.update(1)
        finally:
            for pipe in local_pipes:
                pipe.send(("exit", ()))
            for worker in workers:
                worker.join()

        return BPEParams(vocab=dict(enumerate(vocab)), merges=merges)


    def train_on_strings(self, docs: List[str], n_workers: Optional[int] = None) -> BPEParams:
        # XXX refactor to use inner workings of train_from_file above
        raise NotImplementedError

    
class Tokenizer(ABC):
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        pass
    
    @abstractmethod
    def encode_iterable(self, texts: Iterable[str]) -> Iterator[int]:
        pass

    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        pass

    @abstractmethod
    def is_ending_token(self, token: int) -> bool:
        pass


def encode_doc(args) -> List[int]:
    doc: List[int]
    tokenizer, doc = args
    return tokenizer._encode_single_doc(doc)


class BPETokenizer(Tokenizer):
    def __init__(self, params: BPEParams, special_tokens: Iterable[str] = [], silent: bool = False):
        self.vocab = params.vocab
        self.vocab_lookup: Dict[bytes, int] = {v : k for k, v in self.vocab.items()}
        self.merges = params.merges
        self.token_merges = [(self.vocab_lookup[t1], self.vocab_lookup[t2]) for (t1, t2) in self.merges]
        self.token_merge_lookup = set(self.token_merges)
        self.special_tokens: Set[str] = set(special_tokens)
        self.silent = silent

        assert list(sorted(self.vocab_lookup.values())) == list(range(len(self.vocab_lookup)))

        for token in self.special_tokens:
            token_bytes = bytes(token, encoding="utf-8")
            if token_bytes not in self.vocab_lookup:
                next_token: int = len(self.vocab)
                self.vocab_lookup[token_bytes] = next_token
                self.vocab[next_token] = token_bytes

    @classmethod
    def from_disk(cls, path: os.PathLike) -> Self:
        with open(path, "rb") as f:
            vocab, merges, special_tokens = pickle.load(f)
            return cls(BPEParams(vocab=vocab, merges=merges), special_tokens)

    def to_disk(self, path: os.PathLike) -> None:
        with open(path, "wb") as f:
            pickle.dump((self.vocab, self.merges, self.special_tokens), f)

    def is_ending_token(self, token: int) -> bool:
        return token == self.vocab_lookup[b"<|endoftext|>"]

    def _str_to_raw_tokens(self, text: str) -> Tuple[List[int], List[int]]:
        """
            converts string to raw bytes, first stripping out special tokens and converting those

            returns:
            1. single contiguous list of all tokens
            2. list of document separators for easier splitting and parallelization
        """
        def convert_raw_bytes(s: str) -> List[int]:
            return [self.vocab_lookup[bytes([b])] for b in bytes(s, encoding="utf-8")]

        if len(self.special_tokens) == 0:
            return convert_raw_bytes(text), [0]

        special_token_spans: List[Tuple[int, int]] = []

        for special_token in self.special_tokens:
            special_token_spans.extend(match.span() for match in re.finditer(re.escape(special_token), text))

        # sort by start and then *reverse* end so that we greedily consume the biggest tokens
        # in the case of overlaps
        special_token_spans.sort(key=lambda span: (span[0], -span[1]))

        read_cursor: int = 0
        buffer: List[int] = []
        doc_starts: List[int] = [0]
        with tqdm(total=len(text), unit="MB", unit_scale=True, desc="converting to raw tokens", disable=self.silent) as pbar:
            for start, end in special_token_spans:
                # skip special tokens that were already consumed as subset of previous
                if start < read_cursor:
                    continue

                doc_starts.append(len(buffer))

                # add any raw bytes we've passed since the previous special token
                if start > read_cursor:
                    buffer.extend(convert_raw_bytes(text[read_cursor:start]))

                # slice out the special token itself and add it to the output as a single token
                special_token_bytes: bytes = bytes(text[start:end], encoding="utf-8")
                assert special_token_bytes in self.vocab_lookup, f"unknown special token {text[start:end]}"
                buffer.append(self.vocab_lookup[special_token_bytes])

                read_cursor = end
                pbar.update(read_cursor - pbar.n)

            # add any remaining text after the final special token
            if read_cursor < len(text):
                doc_starts.append(len(buffer))
                buffer.extend(convert_raw_bytes(text[read_cursor:]))

        if not self.silent:
            print(f"converted {len(text)} characters into {len(buffer)} raw tokens")

        return buffer, doc_starts


    def _encode_single_doc(self, doc: List[int]) -> List[int]:
        tokens: List[TokenEntry] = [TokenEntry(token=t, prev=i-1 if i > 0 else None, next=i+1 if i < len(doc) - 1 else None) for i, t in enumerate(doc)]
        pairs: DefaultDict[Tuple[int, int], Set[int]] = defaultdict(set)

        for i, token in enumerate(tqdm(tokens, desc="building initial pair lookup map", disable=self.silent)):
            if token.next is not None:
                pairs[(token.token, tokens[token.next].token)].add(i)

        def add_pair(pair: Tuple[int, int], ix: int) -> None:
            if pair not in self.token_merge_lookup:
                return
            pairs[pair].add(ix)

        def remove_pair(pair: Tuple[int, int], ix: int) -> None:
            if pair not in self.token_merge_lookup:
                return
            pairs[pair].remove(ix)
            if len(pairs[pair]) == 0:
                del pairs[pair]

        for t1, t2 in tqdm(self.token_merges, desc=f"applying merges to {len(tokens)} tokens", disable=self.silent):
            new_token: bytes = self.vocab[t1] + self.vocab[t2]
            assert new_token in self.vocab_lookup
            new_token_id: int = self.vocab_lookup[new_token]

            for pair_index in sorted(pairs[(t1, t2)]):
                """
                    this entry may have already been tombstoned by a previous entry in the list
                    this happens when combining a pair of the same token (e.g. (X, X) -> XX) and that token
                    appears 3+ consecutive times
                """
                if tokens[pair_index].token < 0:
                    continue

                token = tokens[pair_index]
                token.token = new_token_id
                assert token.next is not None
                tokens[token.next].token = -1
                old_token_next: int = token.next
                token.next = tokens[token.next].next

                remove_pair((t1, t2), pair_index)
                if token.prev is not None:
                    remove_pair((tokens[token.prev].token, t1), token.prev)
                    add_pair((tokens[token.prev].token, token.token), token.prev)
                if token.next is not None:
                    tokens[token.next].prev = pair_index
                    remove_pair((t2, tokens[token.next].token), old_token_next)
                    add_pair((token.token, tokens[token.next].token), pair_index)

        return [token.token for token in tokens if token.token >= 0]


    def encode(self, text: str) -> List[int]:
        raw_tokens: List[int]
        doc_starts: List[int]
        raw_tokens, doc_starts = self._str_to_raw_tokens(text)
        doc_chunks = []
        target_chunk_size: int = 1_000_000
        last_doc_start: int = 0
        for doc_start in doc_starts:
            if doc_start - last_doc_start < target_chunk_size:
                continue
            
            doc_chunks.append((self, raw_tokens[last_doc_start:doc_start]))
            last_doc_start = doc_start
        doc_chunks.append((self, raw_tokens[last_doc_start:]))

        encoded_docs: List[List[int]] = Pool().map(encode_doc, doc_chunks)

        return list(chain(*encoded_docs))

        
    def encode_iterable(self, texts: Iterable[str]) -> Iterator[int]:
        # XXX do we need to get more sophisticated than this, or is it safe to assume that
        # every individual string is small enough to safely process in full?
        for text in texts:
            for token in self.encode(text):
                yield token


    def decode(self, ids: List[int]) -> str:
        # XXX probably a more efficient way to do this; could easily parallelize if necessary
        all_bytes: bytes = bytes()
        for token in ids:
            all_bytes += self.vocab[token]
        return all_bytes.decode(encoding="utf-8", errors="ignore")

if __name__ == "__main__":
    import argparse

    eod_token = "<|endoftext|>"

    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["train", "encode"])
    parser.add_argument("text", nargs="?", default="TinyStoriesV2-GPT4-valid.txt")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")

    args = parser.parse_args()

    action: Callable[[], Any]
    if args.action == "train":
        action = lambda: TokenizerTrainer(10_000, {eod_token}).train_on_file(args.text)
    elif args.action == "encode":
        # XXX make tokenizer path configurable
        action = lambda: BPETokenizer.from_disk("tokenizer.pkl", silent=args.quiet).encode(open(args.text, "r").read())

    if args.profile:
        from scalene import scalene_profiler
        scalene_profiler.start()
        with scalene_profiler.enable_profiling():
            action()
        scalene_profiler.stop()
    else:
        action()