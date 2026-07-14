import numpy as np


def validate(sample):
    int_seq = sample["integer_seq"]
    sym_seq = sample["symbol_seq"]
    idx = len(int_seq) - 1
    n_hops = -1
    seq = []
    while True:
        val = int_seq[idx].item()
        seq.append(idx)
        if val == -1:
            break
        idx -= val
        n_hops += 1

    n_hops += 1
    seq[0] = -1

    full_seq = sample["full_seq"]
    reconstructed = np.zeros(len(int_seq), dtype="<U2")
    mask_sym = sym_seq != "-1"
    mask_int = int_seq != -1
    reconstructed[mask_sym] = sym_seq[mask_sym]
    reconstructed[mask_int] = int_seq[mask_int]

    return (
        (idx == sample["integer_id"])
        and (sym_seq[idx] == sample["symbol_id"])
        and (sym_seq[idx] == sample["label"])
        and (n_hops == sample["hop_len"])
        and (np.array(seq) == sample["hop_seq"].numpy()).all()
        and (reconstructed == full_seq).all()
    )
