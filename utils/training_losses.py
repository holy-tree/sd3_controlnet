import warnings

import torch
import torch.nn.functional as F


def select_image_loss_inputs(
    pred_x0: torch.Tensor,
    pixel_values: torch.Tensor,
    image_weights: torch.Tensor,
    max_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Limit differentiable VAE decoding without changing full-batch latent losses."""
    if max_batch_size < 0:
        raise ValueError("image_loss_batch_size must be non-negative")
    batch_size = pred_x0.shape[0]
    if max_batch_size == 0 or max_batch_size >= batch_size:
        return pred_x0, pixel_values, image_weights.flatten()
    return (
        pred_x0[:max_batch_size],
        pixel_values[:max_batch_size],
        image_weights.flatten()[:max_batch_size],
    )


def weighted_spatial_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    beta: float,
    focus_weight: float,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(
            f"Spatial prediction/target shape mismatch: {prediction.shape} != {target.shape}"
        )
    if beta <= 0.0 or focus_weight < 0.0:
        raise ValueError("beta must be positive and focus_weight must be non-negative")
    element_loss = F.smooth_l1_loss(
        prediction,
        target,
        reduction="none",
        beta=beta,
    )
    weights = 1.0 + focus_weight * target
    return (element_loss * weights).sum() / weights.sum().clamp_min(1.0)


def extend_optimizer_state_for_appended_params(
    optimizer: torch.optim.Optimizer,
    state_dict: dict,
    *,
    group_name: str,
    expected_appended_count: int,
) -> dict:
    """Allow old checkpoints to omit parameters appended to one optimizer group."""
    loaded_groups = state_dict.get("param_groups", [])
    if len(loaded_groups) != len(optimizer.param_groups):
        return state_dict
    used_ids = {
        parameter_id
        for group in loaded_groups
        for parameter_id in group.get("params", [])
    }
    next_id = max(used_ids, default=-1) + 1
    for current_group, loaded_group in zip(optimizer.param_groups, loaded_groups):
        if current_group.get("name") != group_name:
            continue
        current_count = len(current_group["params"])
        loaded_count = len(loaded_group.get("params", []))
        if loaded_count >= current_count:
            continue
        missing_count = current_count - loaded_count
        if missing_count != expected_appended_count:
            raise ValueError(
                f"Optimizer group {group_name} differs by {missing_count} parameters; "
                f"only the {expected_appended_count} appended auxiliary-head parameters "
                "can be migrated safely. Use a model sidecar warm start instead of a full "
                "optimizer-state resume for architecture changes."
            )
        loaded_group["params"].extend(range(next_id, next_id + missing_count))
        next_id += missing_count
        warnings.warn(
            f"Optimizer checkpoint is missing {missing_count} appended parameters in "
            f"group {group_name}; they will start with fresh optimizer state.",
            stacklevel=2,
        )
    return state_dict
