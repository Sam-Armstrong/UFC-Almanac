from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Union

from ufc_almanac.globals import STANDARD_TRAINING_DATA_PATH


def load_training_data(
    path: Union[str, Path] = STANDARD_TRAINING_DATA_PATH,
) -> dict[str, torch.Tensor]:
    return torch.load(path, weights_only=True)


def normalize_sequences(
    fighter1: torch.Tensor,
    fighter2: torch.Tensor,
    fighter1_mask: torch.Tensor,
    fighter2_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    combined = torch.cat([fighter1, fighter2], dim=0)
    combined_mask = torch.cat([fighter1_mask, fighter2_mask], dim=0)
    valid_fights = combined[combined_mask.bool()]
    means = valid_fights.mean(dim=0)
    stds = valid_fights.std(dim=0)
    stds[stds == 0] = 1.0
    fighter1 = (fighter1 - means) / stds
    fighter2 = (fighter2 - means) / stds
    return fighter1, fighter2, means, stds


def save_artifacts(
    model: nn.Module,
    model_path: Union[str, Path],
    means: torch.Tensor,
    stds: torch.Tensor,
    normalization_path: Union[str, Path],
) -> None:
    """
    Saves the model and normalization stats to the specified paths.

    Args:
        model: torch.nn.Module
            The model to save.
        model_path: Union[str, Path]
            The path to save the model to.
        means: torch.Tensor
            The means of the features.
        stds: torch.Tensor
            The standard deviations of the features.
        normalization_path: Union[str, Path]
            The path to save the normalization stats to.
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    torch.save(
        {"means": means, "stds": stds},
        normalization_path,
    )
    tqdm.write(f"Saved model to {model_path}")
    tqdm.write(f"Saved normalization stats to {normalization_path}")
