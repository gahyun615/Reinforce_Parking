"""Shared checkpoint metadata helpers."""

import torch


class CheckpointMismatchError(RuntimeError):
    """Raised when checkpoint architecture does not match current environment."""


def infer_obs_dim(ckpt: dict) -> int | None:
    """Infer input dimension from legacy checkpoints without obs_dim metadata."""
    candidates = [
        ("q_net", "net.0.weight"),
        ("actor", "net.0.weight"),
        ("actor", "backbone.0.weight"),
        ("policy", "backbone.0.weight"),
        ("ac", "backbone.0.weight"),
    ]
    for root_key, weight_key in candidates:
        if root_key not in ckpt:
            continue
        state = ckpt[root_key]
        if weight_key in state:
            return int(state[weight_key].shape[1])
    return None


def validate_obs_dim(ckpt: dict, expected_obs_dim: int, algo: str) -> None:
    ckpt_obs_dim = ckpt.get("obs_dim")
    if ckpt_obs_dim is None:
        ckpt_obs_dim = infer_obs_dim(ckpt)

    if ckpt_obs_dim is None:
        return

    if int(ckpt_obs_dim) != int(expected_obs_dim):
        raise CheckpointMismatchError(
            f"{algo.upper()} checkpoint obs_dim={ckpt_obs_dim}, "
            f"but current environment obs_dim={expected_obs_dim}. "
            "Retrain this algorithm with the current environment."
        )
