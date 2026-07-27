from pathlib import Path
import torch
from typing import Any, Optional, Union

from ufc_almanac.globals import NUM_CLASSES


def is_modern_transformer_checkpoint(state_dict: dict[str, torch.Tensor]) -> bool:
    """
    Return True when a checkpoint matches the current transformer architecture.
    """
    return "static_proj.weight" in state_dict


def validate_transformer_checkpoint(state_dict: dict[str, torch.Tensor]) -> None:
    """
    Raise a clear error when a checkpoint cannot be loaded by TransformerModel.
    """
    if is_modern_transformer_checkpoint(state_dict):
        return
    raise ValueError(
        "The checkpoint uses a legacy transformer architecture that is no longer "
        "supported. Retrain with `ufc-train --model transformer`, or pass --path to "
        "a current checkpoint such as artifacts/core/transformer_model.pt."
    )


def infer_num_classes(state_dict: dict[str, torch.Tensor]) -> int:
    """
    Infer the number of output classes from a saved state dict.
    """
    for key in (
        "classifier.3.weight",
        "classifier.0.weight",
        "fc3.weight",
        "linear.weight",
    ):
        if key in state_dict:
            return int(state_dict[key].shape[0])
    return NUM_CLASSES


def infer_transformer_config(state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    """
    Infer transformer architecture from a saved state dict.
    """
    validate_transformer_checkpoint(state_dict)
    d_model = state_dict["static_proj.weight"].shape[0]
    layer_indices = [
        int(key.split(".")[2])
        for key in state_dict
        if key.startswith("transformer.layers.")
    ]
    num_layers = max(layer_indices) + 1 if layer_indices else 2
    return {
        "d_model": d_model,
        "num_layers": num_layers,
    }


def load_model_state_dict(
    model_path: Union[str, Path],
    device: torch.device,
) -> Optional[dict[str, torch.Tensor]]:
    path = Path(model_path)
    if not path.exists():
        return None

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    return None


def load_normalization_artifacts(
    normalization_path: Union[str, Path],
    device: torch.device,
) -> dict[str, Any]:
    path = Path(normalization_path)
    if not path.exists():
        return {}

    artifacts = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(artifacts, dict):
        return {}

    return artifacts
