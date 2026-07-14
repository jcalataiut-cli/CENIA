from src.data.symbolic_dataset import (SequenceSymbolicDatasetGenerator, SequenceSymbolicCollator, SequenceSymbolicTokenizer)
import torch
import numpy as np
import itertools
import random
from pprint import pprint

def generate_symbolic_dataset_collator():

    num_samples = 1200
    vocab_size = 8
    num_leafs = 8
    context_len = 16
    hop_len = [1, 2, 3, 4]


    config = {
        "num_samples": num_samples,
        "num_leafs": num_leafs,
        "vocab_size": vocab_size,
        "context_len": context_len,
        "hop_lens": hop_len,
    }

    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = SequenceSymbolicDatasetGenerator(config)

    tokenizer = SequenceSymbolicTokenizer(vocab_size=vocab_size, num_leafs=num_leafs)
    collator = SequenceSymbolicCollator(tokenizer)

    cdata = collator(dataset)
    
    return cdata

def generate_symbolic_dataset():

    num_samples = 1200
    vocab_size = 24
    num_leafs = 8
    context_len = 16
    hop_len = [1, 2, 3, 4]


    config = {
        "num_samples": num_samples,
        "num_leafs": num_leafs,
        "vocab_size": vocab_size,
        "context_len": context_len,
        "hop_lens": hop_len,
    }

    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = SequenceSymbolicDatasetGenerator(config)
    
    return dataset
