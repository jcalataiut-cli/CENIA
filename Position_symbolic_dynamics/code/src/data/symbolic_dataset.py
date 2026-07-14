import os
import json
import torch
from torch.utils.data import Dataset
import random
import itertools
from collections import Counter
import numpy as np
import string
import os
import json


class SequenceSymbolicDatasetGenerator(Dataset):
    def __init__(self, config):
        self.config = config

        self.num_samples = config["num_samples"]
        self.num_leafs = config["num_leafs"]
        self.vocab_size = config["vocab_size"]
        # vocab is a list of strings (example: ["a", "b", "c"],
        # if vocab_size = 3)
        self.vocab = list(string.ascii_lowercase[: self.vocab_size])
        self.context_len = config["context_len"] + 1
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
            itertools.product(range(self.num_leafs), self.vocab, range(1, self.num_leafs + 1), self.hop_lens)
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

        integer_id, symbol_id, right_value, hop_len = self.triples[idx]
        hop_seq = [integer_id] + sorted(
            random.sample(
                list(np.arange(self.num_leafs, self.context_len - 1)), hop_len - 1
            )
        )
        hop_seq += [-1]
        hop_seq = hop_seq[::-1]
        choices = self.vocab.copy()
        choices.remove(symbol_id)
        hop_symbols = random.sample(choices, k=hop_len - 1)
        hop_symbols += [symbol_id]
        for sym in hop_symbols[:-1]:
            choices.remove(sym)
        symbol_seq = random.choices(choices, k=self.context_len - 1)

        symbol_seq += [random.choice(choices)] #symbol at the end
        symbol_seq[integer_id] = symbol_id
        symbol_seq = np.array(symbol_seq)
        symbol_seq[hop_seq[1:]] = hop_symbols
        right_sequence = np.array(random.choices(choices, k=self.context_len))

        right_sequence[hop_seq[:-1]] = symbol_seq[hop_seq[1:]]
        right_sequence[: self.num_leafs] = random.choices(range(1, self.num_leafs+1), k=self.num_leafs)
        right_sequence[integer_id] = right_value
        hop_seq = torch.tensor(hop_seq)

        # Create a full sequence
        full_seq = np.array(
            [(symbol_seq[i], right_sequence[i]) for i in range(self.context_len)]
        )

        o = {
            "symbol_seq": symbol_seq,
            "right_seq": right_sequence,
            "symbol_id": symbol_id,
            "right_value": right_value,
            "hop_seq": hop_seq,
            "hop_len": hop_len,
            "hop_symbols": hop_symbols,
            "full_seq": full_seq,
        }
        o["label"] = (o["symbol_id"], str(right_value))

        return o


class SequenceSymbolicTokenizer:
    def __init__(self, vocab_size, num_leafs, pad_token_id=0):
        self.vocab_size = vocab_size
        self.num_leafs = num_leafs

        self.pad_token_id = pad_token_id
        self.tokens2ids = {"<PAD>": self.pad_token_id}
        tuples = itertools.product(string.ascii_lowercase[: self.vocab_size], list(range(1, self.num_leafs + 1)))
        
        self.tokens2ids.update(
            {(tuple[0], str(tuple[1])): i
                for i, tuple in enumerate(
                    tuples, start=1)})         
        
        self.tokens2ids.update(
            {
                (letter1, letter2): i
                for i, (letter1, letter2) in enumerate(
                    itertools.product(
                        string.ascii_lowercase[: self.vocab_size],
                        string.ascii_lowercase[: self.vocab_size],
                    ),
                    start=(self.vocab_size * num_leafs) + 1,
                )
            }
        )

        # # Start the last block at the next free id so the maximum id is vocab_size - 1 (zero indexed)
        # self.tokens2ids.update(
        #     {
        #         ("-", letter): i
        #         for i, letter in enumerate(
        #             string.ascii_lowercase[: self.vocab_size],
        #             start=self.vocab_size + 1 + self.vocab_size * self.vocab_size,
        #         )
        #     }
        # )

        #

        self.ids2tokens = {v: k for k, v in self.tokens2ids.items()}

        self.safe_ids2tokens = {v: "".join(k) for k, v in self.tokens2ids.items()}

    def encode(self, sequence):
        """
        sequence: list of str
        """
        if isinstance(sequence, tuple):

            return torch.tensor([self.tokens2ids[sequence]])

        tokenized = torch.tensor([self.tokens2ids[tuple(x)] for x in sequence])
        return tokenized

    def decode(self, token_ids):
        """
        token_ids: list of int
        """
        token_ids_list = [int(x) for x in token_ids]
        decoded = [self.ids2tokens[idx] for idx in token_ids_list]
        return np.array(decoded)

    def safe_decode(self, token_ids):
        """
        token_ids: list of int
        """
        token_ids_list = [int(x) for x in token_ids]
        decoded = [self.safe_ids2tokens[idx] for idx in token_ids_list]
        return np.array(decoded)

    def __len__(self):
        return len(self.tokens2ids)

    def save_pretrained(self, folder_path):
        """
        Saves tokenizer config + vocab to folder_path
        """
        # JSON cannot encode tuple keys, so store a stringified version.
        tokens2ids_json = {str(k): v for k, v in self.tokens2ids.items()}

        data = {
            "pad_token_id": self.pad_token_id,
            "vocab_size": self.vocab_size,
            "tokens2ids": tokens2ids_json,
            "ids2tokens": self.ids2tokens,
        }

        os.makedirs(folder_path, exist_ok=True)
        with open(f"{folder_path}/tokenizer.json", "w") as f:
            json.dump(data, f, indent=2)


class SequenceSymbolicCollator:
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
