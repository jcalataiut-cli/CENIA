"""
Pydantic schemas for experiment configuration.
(Reconstructed based on usage across all command modules)
"""
from pydantic import BaseModel
from typing import List, Optional


class ExperimentConfig(BaseModel):
    """Configuration for multihop and symbolic experiments."""
    exp_name: str
    seed: int = 42
    results_dir: str = "./results"
    bucket_name: str = ""
    save_dataset: bool = False
    eval_on_trainset: bool = False
    
    # Model architecture
    n_layers: int = 12
    n_embd: int = 128
    n_head: int = 8
    rotary_dim: int = 16
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    embed_pdrop: float = 0.1
    tie_word_embeddings: bool = False
    initializer_range: float = 0.02
    nope_from_layer: Optional[int] = None
    
    # Dataset
    num_leafs: int = 8
    vocab_size: int = 120  # Number of letters
    context_len: int = 17
    vocab_size_numbers: Optional[int] = None
    hop_lens: List[int] = [1, 2, 3, 4]
    
    # Training sizes
    n_train: int = 432000
    n_val: int = 2400
    n_test: int = 45600
    
    # Hyperparameters
    epochs: int = 10
    train_batch_size: int = 64
    eval_batch_size: int = 64
    learning_rate: float = 1e-4
    optimizer: str = "adamw_torch"
    adam_beta1: float = 0.95
    adam_beta2: float = 0.999
    weight_decay: float = 1e-4
    
    # Logging/Saving
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 10


class AttentionWeightsConfig(BaseModel):
    """Configuration for attention weights extraction."""
    exp_name: str
    results_dir: str = "./results"
    bucket_name: str = ""
    checkpoints_dir: str = "./checkpoints"
    dataset_path: str = "./dataset.pt"
    min_step: int = 0
    max_step: Optional[int] = None
    delete_after_upload: bool = True


class ComputeMetricsConfig(BaseModel):
    """Configuration for computing positional/symbolic metrics."""
    exp_name: str
    results_dir: str = "./results"
    temp_dir: str = "/tmp"
    bucket_name: str = ""
    bucket_weights_dir: str = ""
    initial_ckpt: int = 0
    final_ckpt: int = 6000
    ckpt_step: int = 500
    query_index: int = 16
    n_samples: int = 100
    delete_downloaded: bool = True


class ComputeMetricsAllHopsConfig(BaseModel):
    """Configuration for metrics with multiple query positions."""
    exp_name: str
    results_dir: str = "./results"
    temp_dir: str = "/tmp"
    bucket_name: str = ""
    bucket_weights_dir: str = ""
    dataset_path: str = "./dataset.pt"
    local_checkpoints_dir: Optional[str] = None
    initial_ckpt: int = 0
    final_ckpt: int = 6000
    ckpt_step: int = 500
    query_type: str = "hops"  # "hops" or "all"
    max_query: Optional[int] = None
    temperatures: List[float] = [1e-4]
    n_samples: int = 100
    delete_downloaded: bool = True
