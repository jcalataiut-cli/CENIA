import os
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F


def get_scores(
    attn_weights_sample: torch.tensor,
    swaps: List[Tuple[int, int]],
    taus: List[float] | None = None,
) -> torch.tensor:
    """Given the attention weights for a sample x, computes the positional
    and symbolic scores for each layer and head. It assumes that the query is the
    last token.

    Args:
        attn_weights_sample (torch.tensor): Attention weights for a sample.
        Shape = (n_layers, n_permutations, n_heads, seq_len, seq_len)
        swaps (List[Tuple[int, int]]): Swaps.
        taus (List[float] | None): List of temperatures. Defaults to None, and in
        that case it will be set to [0.0001].

    Returns:
        torch.tensor: Positional and symbolic scores. Shape = (n_taus, n_layers,
        n_heads, 2). The last dimension contains the positional (0) and symbolic
        (1) score for each layer, head.
    """
    if taus is None:
        taus = [1e-4]
    # Generate tensor with swaps
    swaps_t = torch.tensor(swaps)
    taus_t = torch.tensor(taus)
    base = attn_weights_sample[:, 0, :, -1]  # (n_layers, n_heads, seq_len)
    deltas = abs(
        base[..., swaps_t[:, 0]] - base[..., swaps_t[:, 1]]
    )  # (n_layers, n_heads, n_swaps)
    # weights: (n_taus, n_layers, n_heads, n_swaps)
    weights = F.softmax(deltas / taus_t[..., None, None, None], dim=-1)
    pos_sims = []
    sym_sims = []
    for idx_perm, (iii, jjj) in enumerate(swaps, start=1):
        current_perm = attn_weights_sample[:, idx_perm, :, -1]
        vij = base[..., [iii, jjj]]
        vij_perm = current_perm[..., [iii, jjj]]
        pos_sims.append(F.cosine_similarity(vij_perm, vij, dim=-1))
        sym_sims.append(
            F.cosine_similarity(vij_perm, torch.flip(vij, dims=[-1]), dim=-1)
        )
    pos_sims = torch.stack(pos_sims).permute(1, 2, 0)
    sym_sims = torch.stack(sym_sims).permute(1, 2, 0)

    pos_scores = (weights * pos_sims).sum(axis=-1)
    sym_scores = (weights * sym_sims).sum(axis=-1)

    scores = torch.stack([pos_scores, sym_scores], dim=-1)
    return scores


def get_scores_from_dir(
    folder: Path | str, query_index: int, n_samples: int, taus: List[float] | None
) -> torch.tensor:
    """Given a folder with attention weights, computes the positional and
    symbolic score for each sample.

    Args:
        folder (Path | str): folder with attention weights
        query_index (int): index of the query; used to generate the swaps.
        n_samples (int): number of samples in the folder
        taus (List[float] | None): temperatures.

    Returns:
        torch.tensor: Tensor of shape (n_samples, n_taus, n_layers, n_heads, 2), with
        the positional and symbolic score for each sample, layer and head.
    """
    swaps = generate_swaps(query_index)
    all_scores = []
    for iii in tqdm(range(n_samples), desc="Computing scores...", leave=False):
        path = os.path.join(folder, f"attn_weights_{iii}.pt")
        attw = torch.load(path, map_location=torch.device("cpu"))
        all_scores.append(get_scores(attw, swaps, taus))
    return torch.stack(all_scores)


def generate_swaps(n: int) -> List[Tuple[int, int]]:
    """Generates a list with all the swaps in [0, ..., n - 1]. For example, if
    n = 4, then generate_swaps(n) returns [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]

    Args:
        n (int): n

    Returns:
        List[Tuple[int, int]]: List with swaps
    """
    swaps = []
    for iii in range(n):
        swaps += [(iii, jjj) for jjj in range(iii + 1, n)]
    return swaps


def get_hop_sequence(input_sequence, label):
    """Computes the sequence of hops."""
    current_ptn = len(input_sequence) - 1
    current_value = input_sequence[current_ptn]
    hop_seq = [current_ptn]
    while True:
        current_ptn -= current_value
        current_value = input_sequence[current_ptn]
        if current_value == label:
            return hop_seq[::-1]
        hop_seq.append(current_ptn.detach().item())


def get_scores_multiple_queries(
    queries: List[int],
    attention_weights: torch.tensor,
    max_query: int | None = None,
    taus: List[float] | None = None,
) -> Dict[int, torch.tensor]:
    # max_query is the value used when generating the permutations. If not given,
    # it's deduced from the queries
    all_scores = {}
    if not max_query:
        max_query = max(queries)
    all_swaps = generate_swaps(max_query)
    # We start from 1, because in the attention weights tensor, the first element (index 0) is the unpermuted output.
    # Then the first swap has index 1, the second swap index 2, and so on.
    indexes_all_swaps = {swap: iii for iii, swap in enumerate(all_swaps, start=1)}

    for query in queries:
        swaps_current = generate_swaps(query)
        # We add the first position (index 0), because we need to keep that baseline
        indexes_current = [0] + [indexes_all_swaps[swap] for swap in swaps_current]
        indexes_current = np.array(indexes_current)
        # attention_weights has shape (n_layers, n_permutations, n_heads, seq_len, seq_len)
        # We need to keep:
        # - all the layers (:)
        # - only the permutations needed for the current query (indexes_current)
        # - all the heads (:)
        # - only the weights that the current query could see (query + 1, query + 1)

        attn_current = attention_weights[
            :, indexes_current, :, : query + 1, : query + 1
        ]
        all_scores[query] = get_scores(attn_current, swaps_current, taus)
    return all_scores


def get_scores_multiple_hops_from_dir(
    folder: Path | str,
    queries: List[List[int]],
    n_samples: int,
    max_query: int | None = None,
    taus: List[float] | None = None,
) -> Dict[int, Dict[int, torch.tensor]]:
    """Given a folder with attention weights, computes the positional and
    symbolic score for each sample, using as query index each value in queries.

    Args:
        folder (Path | str): folder with attention weights
        queries (List[List[int]]): list of queries for each sample (len: n_samples)
        n_samples (int): number of samples in the folder
        max_query (int | None): query used when generating the permutations. If not
        given, it will be deduced from the list of hops.
        taus (List[float] | None): temperatures.

    Returns:
        Dict[int, Dict[int, torch.tensor]]: dictionary with scores. The outer key
        is the index of the sample and the value is another dict. The inner dict
        has as key the the query and as value the tensor with scores.
        Each tensor has shape (n_layers, n_heads, 2), and contains the the
        positional and symbolic score (in that order).
    """

    all_scores = {}
    for iii in tqdm(range(n_samples), desc="Computing scores...", leave=False):
        path = os.path.join(folder, f"attn_weights_{iii}.pt")
        attw = torch.load(path, map_location=torch.device("cpu"))
        current_queries = queries[iii]
        all_scores[iii] = get_scores_multiple_queries(
            current_queries, attw, max_query=max_query, taus=taus
        )
    return all_scores
