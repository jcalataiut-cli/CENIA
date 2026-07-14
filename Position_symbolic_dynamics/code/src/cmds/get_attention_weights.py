import os
import json
import glob
import argparse
import logging
import shutil
from typing import List, Tuple
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
import numpy as np
import torch
import boto3
from transformers import GPTJForCausalLM
from .schemas import AttentionWeightsConfig


def build_causal_mask(input_embeds: torch.Tensor):
    """Given a tensor with the input embeddings, creates a causal mask

    Args:
        input_embeds (torch.Tensor): input embeddings. It is assumed that the
        first two dimensions are batch_size and sequence_length.

    Returns:
        torch.Tensor: Attention causal mask, with size (1,1, seq_len, seq_len)
    """
    bsz, seq_len = input_embeds.shape[:2]
    dtype = input_embeds.dtype
    device = input_embeds.device
    causal_mask = torch.full(
        (seq_len, seq_len), fill_value=torch.finfo(dtype).min, device=device
    )
    causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask = causal_mask[None, None, :, :].expand(1, 1, -1, -1)
    return causal_mask


def build_permutations_old(
    hidden_state: torch.Tensor, swaps: List[Tuple[int, int]]
) -> torch.Tensor:
    """Generates all the permutations of a given hidden_state, according to swaps.
    ONLY LEFT HERE FOR LEGACY REASONS.

    Args:
        hidden_state (torch.Tensor): tensor of shape (1, seq_len, model_dim)
        swaps (List[Tuple[int, int]]): List with pairs of swaps. For instance, [(0,1), (0,2),(0,3)]

    Returns:
        torch.Tensor: tensor of shape (len(swaps) + 1, seq_len, model_dim). Each
        element is a permutation, and the first one is the original hidden_state.
    """
    permuted = [hidden_state.clone()]
    for iii, jjj in swaps:
        hidde_state_permuted = hidden_state.clone()
        hidde_state_permuted[:, [iii, jjj], :] = hidde_state_permuted[:, [jjj, iii], :]
        permuted.append(hidde_state_permuted)
    return torch.concat(permuted)


def build_permutations(
    hidden_state: torch.Tensor, swaps: List[Tuple[int, int]]
) -> torch.Tensor:
    """Generates all the permutations of a given hidden_state, according to swaps.
    ONLY LEFT HERE FOR LEGACY REASONS.

    Args:
        hidden_state (torch.Tensor): tensor of shape (1, seq_len, model_dim)
        swaps (List[Tuple[int, int]]): List with pairs of swaps. For instance, [(0,1), (0,2),(0,3)]

    Returns:
        torch.Tensor: tensor of shape (len(swaps) + 1, seq_len, model_dim). Each
        element is a permutation, and the first one is the original hidden_state.
    """
    seq_len = hidden_state.size(1)
    num_swaps = len(swaps)
    device = hidden_state.device

    # 1. Create a base index grid of shape (num_swaps + 1, seq_len)
    # Row 0 is [0, 1, 2...], Row 1 is [0, 1, 2...], etc.
    indices = (
        torch.arange(seq_len, device=device)
        .unsqueeze(0)
        .expand(num_swaps + 1, -1)
        .clone()
    )

    # 2. Convert swaps list to tensors for batch indexing
    # We create a tensor of shape (num_swaps, 2)
    swaps_tensor = torch.tensor(swaps, device=device)

    # Indices for the rows we want to modify (skip row 0, which is the original)
    target_rows = torch.arange(1, num_swaps + 1, device=device)
    col_src = swaps_tensor[:, 0]
    col_dst = swaps_tensor[:, 1]

    # 3. Apply the swaps to the index grid
    # Since indices[row, k] initially equals k, we can just assign the swapped column index.
    indices[target_rows, col_src] = col_dst
    indices[target_rows, col_dst] = col_src

    # 4. Use advanced indexing to gather all permutations at once
    # hidden_state.squeeze(0) has shape (seq_len, model_dim)
    # indices has shape (num_swaps + 1, seq_len)
    # Result has shape (num_swaps + 1, seq_len, model_dim)
    return hidden_state.squeeze(0)[indices]


def get_attention_weights(
    model: GPTJForCausalLM,
    input_ids: torch.Tensor,
    swaps: List[Tuple[int, int]],
    output_dir: Path | str,
) -> None:
    """Compute and save the attention weights for all the inputs, all the swaps,
    all the layers, all the heads.

    For each idx_sample in [0, n_samples - 1], it saves a tensor
    "attn_weights_{idx_sample}.pt". The tensor has shape (n_layers, n_permutations,
    n_heads, seq_len, seq_len).

    Args:
        model (GPTJForCausalLM): model
        input_ids (torch.Tensor): Tensor of shape (n_samples, seq_len), with the
        tokenized inputs
        swaps (List[Tuple[int, int]]): List with pairs of swaps. For instance,
        [(0,1), (0,2),(0,3)]
        output_dir (Path | str): directory where the attention weights will be saved.
    """
    n_samples, seq_len = input_ids.shape
    position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0)
    n_layers = len(model.transformer.h)

    # for idx_sample in tqdm(range(10), desc="Data", leave=False):
    for idx_sample in tqdm(range(n_samples), desc="Data", leave=False):
        current_ids = input_ids[idx_sample].unsqueeze(0)
        output = model(current_ids, output_hidden_states=True, output_attentions=True)
        weights = []
        for n_layer in range(n_layers):
            # if n_layers = 12:
            # output.hidden_states: 0, 1, 2, ..., 12, and 0 is the output of the embedding layer
            prev_hidden_state = output.hidden_states[n_layer].clone()
            all_perms = build_permutations(prev_hidden_state, swaps)
            causal_mask = build_causal_mask(prev_hidden_state)
            _, attn_weights = model.transformer.h[n_layer](
                all_perms,
                position_ids=position_ids,
                attention_mask=causal_mask,
                output_attentions=True,
            )

            # attn_weights.shape = (n_perms, n_heads, seq_len, seq_len)
            # The first permutation is the sequence as it is, so attn_weights[0] should be the same
            # as when we passed the original ids to the model (output.attentions[n_layer][0])
            assert torch.allclose(
                output.attentions[n_layer][0], attn_weights[0], atol=1e-5
            )
            # Just to be sure; the attn_weights of the second element (the first
            # REAL permutation) shouldn't be equal to the original attention weights
            # assert not torch.allclose(
            #     output.attentions[n_layer][0], attn_weights[1], atol=1e-6
            # )
            weights.append(attn_weights)
        weights = torch.stack(weights, dim=0)
        torch.save(weights, output_dir / Path(f"attn_weights_{idx_sample}.pt"))
        # Free GPU memory
        del weights, output
        torch.cuda.empty_cache()


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


def upload_to_s3(s3_client, folder, prefix, bucket):
    for root, dirs, files in os.walk(folder):
        for filename in files:
            full_path = os.path.join(root, filename)
            rel_path = os.path.relpath(full_path, folder)
            s3_key = os.path.join(prefix, os.path.basename(folder), rel_path)

            # print(f"Uploading {full_path} → s3://{self.bucket}/{s3_key}")
            s3_client.upload_file(full_path, bucket, s3_key)


def upload_large_file(
    s3_client, filepath, prefix, bucket, part_size_mb=50, num_threads=32
):
    # Create a TransferConfig object
    transfer_config = boto3.s3.transfer.TransferConfig(
        multipart_threshold=part_size_mb * 1024 * 1024, max_concurrency=num_threads
    )
    # Create an S3 transfer manager
    transfer_manager = boto3.s3.transfer.TransferManager(
        s3_client, config=transfer_config
    )
    _, filename = os.path.split(filepath)
    key = os.path.join(prefix, filename)

    try:
        # Upload the file using multipart upload
        upload = transfer_manager.upload(filepath, bucket, key)
        # Wait for the upload to complete
        upload.result()
        LOGGER.info(f"File uploaded successfully to {bucket}/{key}")

    except Exception as e:
        LOGGER.error(f"Error uploading file: {e}")

    finally:
        # Clean up resources
        transfer_manager.shutdown()


def save_config_and_upload(
    config: AttentionWeightsConfig,
    output_dir: Path | str,
    s3_client,
    bucket: str,
    prefix: Path | str,
):
    path_config = os.path.join(output_dir, "config.json")
    with open(path_config, "w") as fp:
        fp.write(config.model_dump_json(indent=2))
    s3_client.upload_file(path_config, bucket, os.path.join(prefix, "config.json"))


def main(config: AttentionWeightsConfig):
    start = datetime.now()
    now_str = f"{start:%Y-%m-%d_%H-%M-%S}"
    output_dir = Path(config.results_dir) / Path(config.exp_name) / Path(now_str)
    os.makedirs(output_dir, exist_ok=True)

    s3_client = boto3.client("s3")
    s3_prefix = Path(config.exp_name) / Path(now_str)

    save_config_and_upload(
        config,
        output_dir=output_dir,
        s3_client=s3_client,
        bucket=config.bucket_name,
        prefix=str(s3_prefix),
    )

    dataset = torch.load(config.dataset_path)
    input_ids = dataset["input_ids"].to("cuda")
    # The dataset has in the last position the label, and we don't need it
    input_ids = input_ids[:, :-1]

    seq_len = input_ids.shape[-1]
    # We don't want to swap the last position; that's our query
    swaps = generate_swaps(seq_len - 1)

    # We get all the checkpoints dirs
    all_ckpts = glob.glob(os.path.join(config.checkpoints_dir, "checkpoint-*"))
    # The list is lexicographically sorted, and we would like to iterate over numbers
    # in a natural way
    ckpts_steps = sorted([int(path.split("checkpoint-")[-1]) for path in all_ckpts])
    if config.max_step is not None:
        ckpts_steps = [step for step in ckpts_steps if config.min_step < step <= config.max_step]
    # for step in tqdm(ckpts_steps, desc="Checkpoints"):
    LOGGER.info("Starting experiment...")
    for step in tqdm(ckpts_steps, desc="Checkpoints"):
        ckpt_dir = os.path.join(config.checkpoints_dir, f"checkpoint-{step}")
        model = GPTJForCausalLM.from_pretrained(ckpt_dir).to("cuda")
        model.eval()
        current_output_dir = output_dir / Path(f"checkpoint-{step}")
        os.makedirs(current_output_dir, exist_ok=True)
        get_attention_weights(
            model, input_ids=input_ids, swaps=swaps, output_dir=current_output_dir
        )
        LOGGER.info(f"Compressing folder {current_output_dir}")
        # it will return current_output_dir.zip
        compressed_path = shutil.make_archive(
            str(current_output_dir), "zip", current_output_dir
        )
        LOGGER.info(f"Uploading compressed file {compressed_path}")
        upload_large_file(
            s3_client=s3_client,
            filepath=compressed_path,
            prefix=s3_prefix,
            bucket=config.bucket_name,
        )
        # print(f"Uploading {current_output_dir}")
        # upload_to_s3(
        #     s3_client,
        #     folder=current_output_dir,
        #     prefix=s3_prefix,
        #     bucket=config.bucket_name,
        # )
        if config.delete_after_upload:
            os.remove(compressed_path)
            shutil.rmtree(current_output_dir)

        # Free GPU memory
        del model
        torch.cuda.empty_cache()
    end = datetime.now()
    LOGGER.info(f"Experiment successfully completed. Elapsed time: {end - start}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get attention weights experiment")
    parser.add_argument("--config", type=str, default="config.json")

    args = parser.parse_args()
    with open(args.config, "r") as fp:
        dict_config = json.load(fp)

    config = AttentionWeightsConfig.model_validate(dict_config)
    # Config logging
    logging.basicConfig(
        format="{asctime} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M",
        level=logging.INFO,
    )
    LOGGER = logging.getLogger(__name__)
    main(config)
