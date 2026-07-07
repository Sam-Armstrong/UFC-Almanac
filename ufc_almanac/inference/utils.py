from pathlib import Path
import torch
from typing import Any, Optional, Union


def infer_transformer_config(state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    """
    Infer transformer architecture from a saved state dict.
    """
    if "static_proj.weight" in state_dict:
        d_model = state_dict["static_proj.weight"].shape[0]
    else:
        d_model = state_dict["input_proj.weight"].shape[0]
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
