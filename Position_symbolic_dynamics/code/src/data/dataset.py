import os
import json
import torch
import string
from torch.utils.data import Dataset
import random
import itertools
from collections import Counter
import numpy as np

VOCAB = list(string.ascii_lowercase) + [
    string.ascii_letters[iii] + string.ascii_letters[jjj]
    for iii in range(26)
    for jjj in range(26)
]


class SequenceDatasetGenerator(Dataset):
    def __init__(self, config):
        self.config = config

        self.num_samples = config["num_samples"]
        self.num_leafs = config["num_leafs"]
        self.vocab_size = config["vocab_size"]
        self.context_len = config["context_len"]
        self.hop_len = config["hop_len"]
        self.seed = config["seed"] if "seed" in config else None

        if self.seed is not None:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        unique_pairs = list(
            itertools.product(range(self.num_leafs), range(self.vocab_size))
        )
        full_repeats = self.num_samples // len(unique_pairs)
        remainder = self.num_samples % len(unique_pairs)

        self.pairs = unique_pairs * full_repeats + random.sample(
            unique_pairs, remainder
        )
        random.shuffle(self.pairs)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.seed is not None:
            # Deterministic sample-level seed
            seed = self.seed + idx
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)

        integer_id, symbol_id = self.pairs[idx]
        hop_seq = [integer_id] + sorted(
            random.sample(
                list(np.arange(self.num_leafs, self.context_len)), self.hop_len - 1
            )
        )
        hop_seq += [-1]
        hop_seq = hop_seq[::-1]

        symbol_seq = random.choices(list(range(self.vocab_size)), k=self.num_leafs)
        symbol_seq = symbol_seq + [-1] * (self.context_len - self.num_leafs + 1)
        symbol_seq[integer_id] = symbol_id
        symbol_seq = np.array(symbol_seq)

        integer_seq = np.array(
            [0]
            + list(np.random.randint(0, np.arange(1, self.context_len + 1 + 1))[:-1])
        )
        integer_seq[hop_seq[:-1]] = hop_seq[1:]
        integer_seq = np.arange(0, self.context_len + 1) - integer_seq
        integer_seq[: self.num_leafs] = -1

        symbol_seq = torch.tensor(symbol_seq)
        integer_seq = torch.tensor(integer_seq)
        hop_seq = torch.tensor(hop_seq)

        o = {
            "symbol_seq": symbol_seq,
            "integer_seq": integer_seq,
            "integer_id": int(integer_id),
            "symbol_id": int(symbol_id),
            "hop_seq": hop_seq,
        }
        o["label"] = o["symbol_id"]

        return o


def collate_fn(batch):
    symbol_seq = torch.stack([item["symbol_seq"] for item in batch])
    integer_seq = torch.stack([item["integer_seq"] for item in batch])
    hop_seq = torch.stack([item["hop_seq"] for item in batch])

    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    integer_ids = torch.tensor([item["integer_id"] for item in batch], dtype=torch.long)
    symbol_ids = torch.tensor([item["symbol_id"] for item in batch], dtype=torch.long)

    return {
        "symbol_seq": symbol_seq,
        "integer_seq": integer_seq,
        "hop_seq": hop_seq,
        "labels": labels,
        "js": integer_ids,
        "symbol_ids": symbol_ids,
    }


class SequenceMultiHopDatasetGenerator(Dataset):
    def __init__(self, config):
        self.config = config

        self.num_samples = config["num_samples"]
        self.num_leafs = config["num_leafs"]
        self.vocab_size = config["vocab_size"]
        # vocab is a list of strings (example: ["a", "b", "c"],
        # if vocab_size = 3)
        self.vocab = VOCAB[: self.vocab_size]
        self.context_len = config["context_len"]
        # hop_lens is a list (example: [1,2,3,4])
        self.hop_lens = config["hop_lens"]  # list of possible lens
        self.seed = config["seed"] if "seed" in config else None

        if self.seed is not None:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
        # To enforce that all the hops are equally distributed, we create
        # triplets, analog to the creation of pairs in the previous dataset
        unique_triples = list(
            itertools.product(range(self.num_leafs), self.vocab, self.hop_lens)
        )

        full_repeats = self.num_samples // len(unique_triples)
        remainder = self.num_samples % len(unique_triples)

        self.triples = unique_triples * full_repeats + random.sample(
            unique_triples, remainder
        )
        random.shuffle(self.triples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.seed is not None:
            # Deterministic sample-level seed
            seed = self.seed + idx
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)

        integer_id, symbol_id, hop_len = self.triples[idx]
        hop_seq = [integer_id] + sorted(
            random.sample(
                list(np.arange(self.num_leafs, self.context_len)), hop_len - 1
            )
        )
        hop_seq += [-1]
        hop_seq = hop_seq[::-1]

        symbol_seq = random.choices(self.vocab, k=self.num_leafs)
        symbol_seq = symbol_seq + [-1] * (self.context_len - self.num_leafs + 1)
        symbol_seq[integer_id] = symbol_id
        symbol_seq = np.array(symbol_seq)

        integer_seq = np.array(
            [0]
            + list(np.random.randint(0, np.arange(1, self.context_len + 1 + 1))[:-1])
        )
        integer_seq[hop_seq[:-1]] = hop_seq[1:]
        integer_seq = np.arange(0, self.context_len + 1) - integer_seq
        integer_seq[: self.num_leafs] = -1

        symbol_seq = symbol_seq
        integer_seq = torch.tensor(integer_seq)
        hop_seq = torch.tensor(hop_seq)

        # Create a full sequence
        full_seq = np.zeros(len(symbol_seq), dtype="<U2")
        mask_sym = symbol_seq != "-1"
        mask_int = integer_seq != -1
        full_seq[mask_sym] = symbol_seq[mask_sym]
        full_seq[mask_int] = integer_seq[mask_int]

        o = {
            "symbol_seq": symbol_seq,
            "integer_seq": integer_seq,
            "integer_id": int(integer_id),
            "symbol_id": symbol_id,
            "hop_seq": hop_seq,
            "hop_len": hop_len,
            "full_seq": full_seq,
        }
        o["label"] = o["symbol_id"]

        return o


class SequenceMultiHopTokenizer:
    def __init__(self, vocab_size_numbers, vocab_size_str, pad_token_id=0):
        self.vocab_size_numbers = vocab_size_numbers
        self.vocab_size_str = vocab_size_str

        self.pad_token_id = pad_token_id
        self.tokens2ids = {"<PAD>": self.pad_token_id}
        self.tokens2ids.update(
            {str(i): i for i in range(1, self.vocab_size_numbers + 1)}
        )
        self.tokens2ids.update(
            {
                letter: i
                for i, letter in enumerate(
                    VOCAB[: self.vocab_size_str],
                    start=self.vocab_size_numbers + 1,
                )
            }
        )

        self.ids2tokens = {v: k for k, v in self.tokens2ids.items()}

    def encode(self, sequence):
        """
        sequence: list of str
        """
        if type(sequence) == str:
            sequence = [sequence]
        tokenized = torch.tensor([self.tokens2ids[x] for x in sequence])
        return tokenized

    def decode(self, token_ids):
        """
        token_ids: list of int
        """
        token_ids_list = [int(x) for x in token_ids]
        decoded = [self.ids2tokens[idx] for idx in token_ids_list]
        return np.array(decoded)

    def __len__(self):
        return len(self.tokens2ids)

    def save_pretrained(self, folder_path):
        """
        Saves tokenizer config + vocab to folder_path
        """
        data = {
            "pad_token_id": self.pad_token_id,
            "vocab_size_numbers": self.vocab_size_numbers,
            "vocab_size_str": self.vocab_size_str,
            "tokens2ids": self.tokens2ids,
            "ids2tokens": self.ids2tokens,
        }

        os.makedirs(folder_path, exist_ok=True)
        with open(f"{folder_path}/tokenizer.json", "w") as f:
            json.dump(data, f, indent=2)


class SequenceMultiHopCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        tokenized = torch.stack(
            [self.tokenizer.encode(item["full_seq"]) for item in batch]
        )
        labels_simple = torch.tensor(
            [self.tokenizer.encode(item["label"]) for item in batch]
        )
        # Concatenate the input and the labels
        full_seq = torch.cat([tokenized, labels_simple.unsqueeze(1)], dim=1)

        labels = torch.full_like(full_seq, fill_value=-100)
        labels[:, -1] = labels_simple

        hop_lens = torch.tensor([item["hop_len"] for item in batch])

        # We only care about the last token, so we set everything as pad, except the last position
        # labels = torch.full_like(tokenized, fill_value=-100)

        # Unncomment the following lines if you want a fully autoregressive model
        # labels = tokenized.clone()
        # labels[:, :-1] = labels[:, 1:]
        # labels[:, -1] = labels_simple
        batch_out = {
            "input_ids": full_seq,
            "labels": labels,  # the trainer will shift the labels
            "hop_lens": hop_lens,  # Keep it in the batch
        }

        return batch_out

    def save_dataset(self, batch, path):
        batch_collated = self.__call__(batch)
        torch.save(batch_collated, path)
