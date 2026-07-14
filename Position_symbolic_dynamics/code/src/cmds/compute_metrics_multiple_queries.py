import os
import json
import shutil
import logging
import argparse
from datetime import datetime
import torch
from tqdm import tqdm
from ..lib import metrics, s3
from .schemas import ComputeMetricsAllHopsConfig


def main(config: ComputeMetricsAllHopsConfig):
    start = datetime.now()
    now_str = f"{start:%Y-%m-%d_%H-%M-%S}"
    tmp_dir = os.path.join(config.temp_dir, now_str)
    results_dir = os.path.join(config.results_dir, config.exp_name, now_str)
    bucket_prefix = os.path.join(config.exp_name, now_str)
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    s3.save_config_and_upload(config, results_dir, config.bucket_name, bucket_prefix)
    s3.upload_file(config.dataset_path, config.bucket_name, bucket_prefix)

    dataset = torch.load(config.dataset_path)
    input_ids = dataset["input_ids"][:, :-1]
    labels = dataset["input_ids"][:, -1]

    if config.local_checkpoints_dir:
        if not os.path.isdir(config.local_checkpoints_dir):
            raise FileNotFoundError(
                f"Local checkpoints dir not found: {config.local_checkpoints_dir}"
            )
        LOGGER.info(
            f"Using local checkpoints from {config.local_checkpoints_dir}"
        )

    n_samples = len(input_ids)
    if config.query_type == "hops":
        queries = [
            metrics.get_hop_sequence(seq, label)
            for seq, label in zip(input_ids, labels)
        ]

        hop_lens_comp = [len(hop) for hop in queries]
        assert (torch.tensor(hop_lens_comp) == dataset["hop_lens"]).all()

    else:  # config.query_type == "all"
        max_index = config.max_query + 1 if config.max_query else input_ids.shape[1]
        # we start from 2, because is the first index where there can be a swap
        queries = torch.arange(2, max_index).expand(input_ids.shape[0], -1).tolist()

    for ckpt_step in tqdm(
        range(config.initial_ckpt, config.final_ckpt + 1, config.ckpt_step),
        desc="Checkpoints...",
    ):
        cleanup_paths = []
        if config.local_checkpoints_dir:
            local_folder = os.path.join(
                config.local_checkpoints_dir, f"checkpoint-{ckpt_step}"
            )
            local_zip = os.path.join(
                config.local_checkpoints_dir, f"checkpoint-{ckpt_step}.zip"
            )
            if os.path.isdir(local_folder):
                LOGGER.info(
                    f"Using local checkpoint directory {local_folder}"
                )
                tmp_local_folder = local_folder
            elif os.path.isfile(local_zip):
                tmp_local_folder = os.path.join(tmp_dir, f"checkpoint-{ckpt_step}")
                LOGGER.info(f"Unzipping local file {local_zip}")
                shutil.unpack_archive(local_zip, tmp_local_folder, "zip")
                cleanup_paths.append(tmp_local_folder)
            else:
                raise FileNotFoundError(
                    "Missing local checkpoint: expected "
                    f"{local_folder} or {local_zip}"
                )
        else:
            current_key = os.path.join(
                config.bucket_weights_dir, f"checkpoint-{ckpt_step}.zip"
            )
            tmp_local_path = os.path.join(tmp_dir, f"checkpoint-{ckpt_step}.zip")
            LOGGER.info(f"Downloading file {current_key}...")
            s3.download_large_file(config.bucket_name, current_key, tmp_local_path)
            tmp_local_folder = os.path.join(tmp_dir, f"checkpoint-{ckpt_step}")
            LOGGER.info(f"Unzipping file {tmp_local_path}")
            shutil.unpack_archive(tmp_local_path, tmp_local_folder, "zip")
            cleanup_paths.extend([tmp_local_path, tmp_local_folder])

        scores = metrics.get_scores_multiple_hops_from_dir(
            tmp_local_folder,
            queries,
            n_samples,
            max_query=config.max_query,
            taus=config.temperatures,
        )

        scores_path = os.path.join(results_dir, f"checkpoint-{ckpt_step}.pt")
        torch.save(scores, scores_path)

        if config.delete_downloaded:
            for path in cleanup_paths:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                elif os.path.isfile(path):
                    os.remove(path)

        s3.upload_file(scores_path, config.bucket_name, bucket_prefix)
    end = datetime.now()
    LOGGER.info(
        f"Metrics computed. Results saved in {results_dir} Elapsed time: {end - start}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute metrics from attention weights"
    )
    parser.add_argument("--config", type=str, default="config.json")

    args = parser.parse_args()
    with open(args.config, "r") as fp:
        dict_config = json.load(fp)

    config = ComputeMetricsAllHopsConfig.model_validate(dict_config)
    # Config logging
    logging.basicConfig(
        format="{asctime} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M",
        level=logging.INFO,
    )
    LOGGER = logging.getLogger(__name__)
    main(config)
