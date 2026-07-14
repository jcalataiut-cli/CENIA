import os
from pathlib import Path
from pydantic import BaseModel
import boto3
from boto3.s3.transfer import TransferConfig


def download_large_file(bucket, key, local_path, max_concurrency=20):
    s3_client = boto3.client("s3")

    config = TransferConfig(
        multipart_threshold=1024 * 1024 * 50,  # 50MB: threshold for multipart
        multipart_chunksize=1024 * 1024 * 50,  # 50MB part size
        max_concurrency=max_concurrency,  # parallel threads
        use_threads=True,
    )

    s3_client.download_file(Bucket=bucket, Key=key, Filename=local_path, Config=config)


def save_config_and_upload(
    config: BaseModel,
    output_dir: Path | str,
    bucket: str,
    prefix: Path | str,
):
    s3_client = boto3.client("s3")
    path_config = os.path.join(output_dir, "config.json")
    with open(path_config, "w") as fp:
        fp.write(config.model_dump_json(indent=2))
    s3_client.upload_file(path_config, bucket, os.path.join(prefix, "config.json"))


def upload_file(filepath, bucket, prefix):
    filename = os.path.split(filepath)[-1]
    s3_client = boto3.client("s3")
    s3_client.upload_file(filepath, bucket, os.path.join(prefix, filename))
