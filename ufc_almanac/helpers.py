from pathlib import Path
import torch
from typing import Optional, Union

from ufc_almanac.globals import CHECKPOINTS_DIR


def get_device() -> torch.device:
    """
    Get the device to use for training and inference.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def resolve_checkpoint_paths(
    model: type[torch.nn.Module],
    model_path: Optional[Union[str, Path]] = None,
) -> tuple[Path, Path]:
    """
    Resolve checkpoint paths for saving or loading a trained model.
    """
    model_name = model.__name__
    resolved_model_path = (
        Path(model_path)
        if model_path is not None
        else Path(CHECKPOINTS_DIR) / f"{model_name}.pt"
    )
    resolved_normalization_path = resolved_model_path.with_name(
        f"{resolved_model_path.stem}_normalization{resolved_model_path.suffix}"
    )
    return resolved_model_path, resolved_normalization_path

def resolve_model(
    model_name: str,
    models: dict[str, torch.nn.Module],
) -> torch.nn.Module:
    """
    Resolve the model class from the model name.
    """
    if model_name in models:
        return models[model_name]
    else:
        for model_class in models.values():
            if model_class.__name__ == model_name:
                return model_class

        raise ValueError(
            f"Unknown model {model_name!r}. "
            f"Available models: {', '.join(sorted(models))}"
        )
