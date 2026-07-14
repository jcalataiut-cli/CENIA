import os
import json
import random
import argparse
import numpy as np
import torch
from transformers import GPTJConfig, GPTJForCausalLM, TrainingArguments
from transformers.models.gptj import modeling_gptj
from ..data.symbolic_dataset import (
    SequenceSymbolicTokenizer,
    SequenceSymbolicDatasetGenerator,
    SequenceSymbolicCollator,
)
from datetime import datetime
from .schemas import ExperimentConfig
from pathlib import Path
from ..models.train import (
    ExtraParametersTrainer,
    compute_metrics_with_hops,
    UploadToS3Callback,
)
from ..lib import s3


def set_seeds(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)


def monkey_patch_apply_rotary_pos_emb(
    tensor: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor
) -> torch.Tensor:
    # When using Nope, the shape of the tensor is 0. In that case, return the tensor as it is.
    # That's the only change.
    if tensor.shape[-1] == 0:
        return tensor
    sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    return (tensor * cos) + (modeling_gptj.rotate_every_two(tensor) * sin)


def main(config: ExperimentConfig):
    # Monkey patch
    modeling_gptj.apply_rotary_pos_emb = monkey_patch_apply_rotary_pos_emb
    now_str = f"{datetime.now():%Y-%m-%d_%H-%M-%S}"
    set_seeds(config.seed)
    output_dir = Path(config.results_dir) / Path(config.exp_name) / Path(now_str)
    os.makedirs(output_dir, exist_ok=True)
    s3_prefix = Path(config.exp_name) / Path(now_str)
    s3.save_config_and_upload(config, output_dir, config.bucket_name, s3_prefix)

    tokenizer = SequenceSymbolicTokenizer(vocab_size=config.vocab_size, num_leafs=config.num_leafs)
    collator = SequenceSymbolicCollator(tokenizer)

    gpt_config = GPTJConfig(
        vocab_size=len(tokenizer),
        n_positions=128,
        n_embd=config.n_embd,  # Residual stream dimension
        n_layer=config.n_layers,  # 12 layers
        n_head=config.n_head,  # 8 attention heads
        rotary_dim=config.rotary_dim,  # RoPE dimension (matches head dim)
        n_inner=None,  # Defaults to 4 * n_embd (for GELU MLP)
        activation_function="gelu",  # GELU activation
        resid_pdrop=0.1,  # Dropout rate 0.1
        attn_pdrop=0.1,  # Dropout rate 0.1
        embed_pdrop=0.1,  # Dropout rate 0.1
        layer_norm_epsilon=1e-5,
        tie_word_embeddings=config.tie_word_embeddings,  # Do not tie input/output embeddings
        pad_token_id=tokenizer.pad_token_id,
        use_cache=False,
    )

    model = GPTJForCausalLM(gpt_config)
    if config.nope_from_layer is not None:
        for idx_layer in range(config.nope_from_layer, config.n_layers):
            model.transformer.h[idx_layer].attn.rotary_dim = 0

    num_params = model.num_parameters()
    print(f"Model configured. Total parameters: {num_params / 1_000_000:.2f}M")

    # Dataset generation
    dataset_config_base = {
        "num_leafs": config.num_leafs,
        "vocab_size": config.vocab_size,
        "context_len": config.context_len,
        "seed": config.seed,
        "hop_lens": config.hop_lens,
    }

    train_config = dataset_config_base.copy()
    train_config["num_samples"] = config.n_train
    train_dataset = SequenceSymbolicDatasetGenerator(train_config)

    val_config = dataset_config_base.copy()
    val_config["seed"] += config.n_train  # increase seed
    val_config["num_samples"] = config.n_val
    val_dataset = SequenceSymbolicDatasetGenerator(val_config)

    test_config = dataset_config_base.copy()
    test_config["seed"] += config.n_train + config.n_val
    test_config["num_samples"] = config.n_test
    test_dataset = SequenceSymbolicDatasetGenerator(test_config)

    if config.save_dataset:
        print("Saving train dataset...")
        path_train = os.path.join(output_dir, "train_dataset.pt")
        collator.save_dataset(train_dataset, path_train)
        print("Uploading train dataset...")
        s3.upload_file(path_train, config.bucket_name, s3_prefix)

        print("Saving val dataset...")
        path_val = os.path.join(output_dir, "val_dataset.pt")
        collator.save_dataset(val_dataset, path_val)
        print("Uploading val dataset...")
        s3.upload_file(path_val, config.bucket_name, s3_prefix)

        print("Saving test dataset...")
        path_test = os.path.join(output_dir, "test_dataset.pt")
        collator.save_dataset(test_dataset, path_test)
        print("Uploading test dataset...")
        s3.upload_file(path_test, config.bucket_name, s3_prefix)

    if config.eval_on_trainset:
        trainer_evalset = {"train": train_dataset, "val": val_dataset}
        metric_best_model = "eval_val_accuracy"
    else:
        trainer_evalset = val_dataset
        metric_best_model = "eval_accuracy"

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        # --- Epochs, Batch Size, LR ---
        num_train_epochs=config.epochs,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        learning_rate=1e-4,
        # --- Optimizer (AdamW) ---
        optim="adamw_torch",
        adam_beta1=0.95,  # From paper
        adam_beta2=0.999,  # From paper
        # --- Scheduler (Linear w/ Warmup) ---
        lr_scheduler_type="linear",
        warmup_steps=750,  # From paper
        # --- Regularization ---
        weight_decay=1e-4,  # From paper
        # --- Logging and Saving ---
        logging_strategy="steps",
        logging_steps=config.eval_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model=metric_best_model,
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
    )

    trainer = ExtraParametersTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=trainer_evalset,
        data_collator=collator,
        compute_metrics=compute_metrics_with_hops,
        callbacks=[UploadToS3Callback(config.bucket_name, s3_prefix)],
    )

    trainer.train()

    print("Best model saved to:", trainer.state.best_model_checkpoint)

    test_results = trainer.evaluate(test_dataset)
    print(f"Accuracy: {test_results['eval_accuracy'] * 100:.3f}%")
    print(f"Loss: {test_results['eval_loss']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Symbolic experiment")
    parser.add_argument("--config", type=str, default="config.json")

    args = parser.parse_args()
    with open(args.config, "r") as fp:
        dict_config = json.load(fp)

    config = ExperimentConfig.model_validate(dict_config)

    main(config)
