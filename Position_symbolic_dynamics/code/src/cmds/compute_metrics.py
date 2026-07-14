import os
import json
import shutil
import logging
import argparse
from datetime import datetime
import torch
from tqdm import tqdm
from ..lib import metrics, s3
from .schemas import ComputeMetricsConfig


def main(config: ComputeMetricsConfig):
    start = datetime.now()
    now_str = f"{start:%Y-%m-%d_%H-%M-%S}"
    tmp_dir = os.path.join(config.temp_dir, now_str)
    results_dir = os.path.join(config.results_dir, config.exp_name, now_str)
    bucket_prefix = os.path.join(config.exp_name, now_str)
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    s3.save_config_and_upload(config, results_dir, config.bucket_name, bucket_prefix)
    for ckpt_step in tqdm(
        range(config.initial_ckpt, config.final_ckpt + 1, config.ckpt_step),
        desc="Checkpoints...",
    ):
        current_key = os.path.join(
            config.bucket_weights_dir, f"checkpoint-{ckpt_step}.zip"
        )
        tmp_local_path = os.path.join(tmp_dir, f"checkpoint-{ckpt_step}.zip")
        LOGGER.info(f"Downloeading file {current_key}...")
        s3.download_large_file(config.bucket_name, current_key, tmp_local_path)
        tmp_local_folder = os.path.join(tmp_dir, f"checkpoint-{ckpt_step}")
        LOGGER.info(f"Unzipping file {tmp_local_path}")
        shutil.unpack_archive(tmp_local_path, tmp_local_folder, "zip")
        scores = metrics.get_scores_from_dir(
            tmp_local_folder, config.query_index, config.n_samples
        )
        scores_mean = scores.mean(axis=0)

        scores_path = os.path.join(results_dir, f"checkpoint-{ckpt_step}-all.pt")
        scores_mean_path = os.path.join(results_dir, f"checkpoint-{ckpt_step}-mean.pt")
        torch.save(scores, scores_path)
        torch.save(scores_mean, scores_mean_path)
        if config.delete_downloaded:
            os.remove(tmp_local_path)
            shutil.rmtree(tmp_local_folder)

        s3.upload_file(scores_path, config.bucket_name, bucket_prefix)
        s3.upload_file(scores_mean_path, config.bucket_name, bucket_prefix)
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

    config = ComputeMetricsConfig.model_validate(dict_config)
    # Config logging
    logging.basicConfig(
        format="{asctime} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M",
        level=logging.INFO,
    )
    LOGGER = logging.getLogger(__name__)
    main(config)
