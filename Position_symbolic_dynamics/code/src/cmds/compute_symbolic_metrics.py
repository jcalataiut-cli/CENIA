import os
import json
import shutil
import logging
import argparse
from datetime import datetime
import torch
from tqdm import tqdm
from generate_dataset_sym import generate_symbolic_dataset, generate_symbolic_dataset_collator
from lib import metrics, s3
from schemas import ComputeMetricsAllHopsConfig


def main(config: ComputeMetricsAllHopsConfig):
    start = datetime.now()
    now_str = f"{start:%Y-%m-%d_%H-%M-%S}"
    tmp_dir = os.path.join(config.temp_dir, now_str)
    results_dir = os.path.join(config.results_dir, config.exp_name, now_str)
    bucket_prefix = os.path.join(config.exp_name, now_str)
    weights_dir = os.path.join(config.bucket_weights_dir, config.model_date)

    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    s3.save_config_and_upload(config, results_dir, config.bucket_name, bucket_prefix)
    s3.upload_file(config.dataset_path, config.bucket_name, bucket_prefix)

    dataset = torch.load(config.dataset_path)
    #dataset = generate_symbolic_dataset_collator()
    input_ids = dataset["input_ids"][:, :-1]
    labels = dataset["input_ids"][:, -1]

    n_samples = len(input_ids)
    #dataset_total = generate_symbolic_dataset()
    sample_lens = [
       len(item) for item in input_ids
    ]

    #hop_lens_comp = [len(hop) - 1 for hop in hops]
    #assert (torch.tensor(hop_lens_comp) == dataset["hop_lens"]).all()

    for ckpt_step in tqdm(
        range(config.initial_ckpt, config.final_ckpt + 1, config.ckpt_step),
        desc="Checkpoints...",
    ):
        current_key = os.path.join(
            config.bucket_weights_dir, config.model_date, f"checkpoint-{ckpt_step}.zip"
        )
        tmp_local_path = os.path.join(tmp_dir, f"checkpoint-{ckpt_step}.zip")
        LOGGER.info(f"Downloeading file {current_key}...")
        s3.download_large_file(config.bucket_name, current_key, tmp_local_path)
        tmp_local_folder = os.path.join(tmp_dir, f"checkpoint-{ckpt_step}")
        LOGGER.info(f"Unzipping file {tmp_local_path}")
        shutil.unpack_archive(tmp_local_path, tmp_local_folder, "zip")

        scores = metrics.get_scores_multiple_all_from_dir(
            tmp_local_folder, sample_lens, n_samples, max_query=config.max_query
        )

        scores_path = os.path.join(results_dir, f"checkpoint-{ckpt_step}.pt")
        torch.save(scores, scores_path)

        if config.delete_downloaded:
            os.remove(tmp_local_path)
            shutil.rmtree(tmp_local_folder)

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
