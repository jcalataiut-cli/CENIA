"""
Custom Trainer for multi-hop experiments with last-token prediction.

Based on HuggingFace Trainer with modifications:
- ExtraParametersTrainer: passes hop_lens through the batch
- compute_metrics_with_hops: accuracy per hop condition
- UploadToS3Callback: uploads checkpoints to S3
"""

import os
import torch
import numpy as np
from transformers import Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback
from typing import Dict, Optional


class ExtraParametersTrainer(Trainer):
    """
    Trainer that preserves extra batch parameters (hop_lens)
    during evaluation.
    """
    
    def prediction_step(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: Optional[list] = None,
    ):
        """
        Custom prediction step that handles hop_lens.
        """
        labels = inputs.get("labels")
        hop_lens = inputs.pop("hop_lens", None)
        
        with torch.no_grad():
            loss, logits, _ = super().prediction_step(
                model, inputs, prediction_loss_only=False, ignore_keys=ignore_keys
            )
        
        if hop_lens is not None:
            return (loss, logits, labels, hop_lens)
        return (loss, logits, labels)
    
    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[list] = None,
        metric_key_prefix: str = "eval",
    ):
        """
        Custom evaluation loop that passes hop_lens to compute_metrics.
        """
        # Store hop_lens from batches
        all_hop_lens = []
        all_labels = []
        all_logits = []
        
        for batch in dataloader:
            hop_lens = batch.get("hop_lens")
            if hop_lens is not None:
                all_hop_lens.append(hop_lens.cpu())
            
            with torch.no_grad():
                outputs = model(
                    input_ids=batch["input_ids"].to(self.args.device),
                    labels=batch["labels"].to(self.args.device),
                )
            
            all_logits.append(outputs.logits.cpu())
            all_labels.append(batch["labels"].cpu())
        
        # Concatenate all batches
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Compute metrics
        metrics = {}
        if all_hop_lens:
            all_hop_lens = torch.cat(all_hop_lens, dim=0)
            predictions = all_logits[:, -1, :].argmax(dim=-1)
            targets = all_labels[:, -1]
            
            # Global accuracy
            correct = (predictions == targets).sum().item()
            total = len(targets)
            metrics[f"{metric_key_prefix}_accuracy"] = correct / total
            
            # Per-hop accuracy
            unique_hops = torch.unique(all_hop_lens)
            for hop in unique_hops:
                mask = all_hop_lens == hop
                hop_correct = (predictions[mask] == targets[mask]).sum().item()
                hop_total = mask.sum().item()
                metrics[f"{metric_key_prefix}_hop_{hop}_accuracy"] = (
                    hop_correct / hop_total if hop_total > 0 else 0.0
                )
        
        # Loss
        metrics[f"{metric_key_prefix}_loss"] = outputs.loss.item() if hasattr(outputs, 'loss') else 0.0
        
        return metrics


def compute_metrics_with_hops(eval_pred):
    """
    Compute accuracy metrics with hop breakdown.
    
    Args:
        eval_pred: EvalPrediction with predictions, label_ids, and hop_lens
    
    Returns:
        dict with accuracy and per-hop accuracies
    """
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    
    predictions = logits[:, -1, :].argmax(axis=-1)
    targets = labels[:, -1]
    
    # Global accuracy
    correct = (predictions == targets).sum()
    total = len(targets)
    metrics = {"accuracy": correct / total}
    
    # If hop_lens is available (passed as third element)
    if len(eval_pred) > 2 and eval_pred[2] is not None:
        hop_lens = eval_pred[2]
        unique_hops = np.unique(hop_lens)
        for hop in unique_hops:
            mask = hop_lens == hop
            hop_correct = (predictions[mask] == targets[mask]).sum()
            hop_total = mask.sum()
            metrics[f"hop_{hop}_accuracy"] = (
                hop_correct / hop_total if hop_total > 0 else 0.0
            )
    
    return metrics


class UploadToS3Callback(TrainerCallback):
    """
    Callback that uploads checkpoints to S3.
    """
    
    def __init__(self, bucket_name: str, prefix: str):
        self.bucket_name = bucket_name
        self.prefix = prefix
    
    def on_save(self, args, state, control, **kwargs):
        """
        Called when a checkpoint is saved.
        Uploads the checkpoint directory to S3.
        """
        if not self.bucket_name:
            return
        
        import boto3
        import shutil
        
        s3_client = boto3.client("s3")
        output_dir = args.output_dir
        checkpoint_dir = os.path.join(
            output_dir, f"checkpoint-{state.global_step}"
        )
        
        if os.path.isdir(checkpoint_dir):
            # Compress and upload
            archive_path = shutil.make_archive(
                checkpoint_dir, "zip", checkpoint_dir
            )
            s3_key = os.path.join(self.prefix, os.path.basename(archive_path))
            s3_client.upload_file(archive_path, self.bucket_name, s3_key)
            os.remove(archive_path)
