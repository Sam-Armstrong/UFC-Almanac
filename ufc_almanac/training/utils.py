from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Any, Union

from ufc_almanac.globals import STANDARD_TRAINING_DATA_PATH, VERBOSE


def collect_validation_logits(
    model: nn.Module,
    device: torch.device,
    data_loader: DataLoader,
    is_transformer: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collect logits and labels from a validation loader for post-hoc calibration.
    """
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        if is_transformer:
            for batch in data_loader:
                (
                    fighter1,
                    fighter2,
                    mask1,
                    mask2,
                    days_before1,
                    days_before2,
                    days_gap1,
                    days_gap2,
                    matchup_features,
                    labels,
                ) = [tensor.to(device) for tensor in batch]
                logits = model(
                    fighter1,
                    fighter2,
                    mask1,
                    mask2,
                    days_before1,
                    days_before2,
                    days_gap1,
                    days_gap2,
                    matchup_features,
                )
                all_logits.append(logits)
                all_labels.append(labels)
        else:
            for batch_features, batch_labels in data_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                logits = model(batch_features)
                all_logits.append(logits)
                all_labels.append(batch_labels)

    return torch.cat(all_logits), torch.cat(all_labels)

def load_training_data(
    path: Union[str, Path] = STANDARD_TRAINING_DATA_PATH,
) -> dict[str, torch.Tensor]:
    return torch.load(path, weights_only=True)

def temporal_train_val_split(
    num_samples: int,
    val_fraction: float,
    fight_dates: torch.Tensor | None = None,
) -> tuple[list[int], list[int]]:
    """
    Split sample indices into train and validation sets.

    Validation is the n most recent samples by fight date, where
    n = max(1, int(num_samples * val_fraction)).
    """
    val_size = max(1, int(num_samples * val_fraction))
    if fight_dates is not None:
        sorted_indices = fight_dates.argsort(stable=True).tolist()
    else:
        sorted_indices = list(range(num_samples))
    train_indices = sorted_indices[:-val_size]
    val_indices = sorted_indices[-val_size:]
    return train_indices, val_indices

def extract_model_config(model: nn.Module) -> dict[str, Any]:
    """
    Capture constructor kwargs needed to reload a trained model.
    """
    model_name = model.__class__.__name__
    if model_name == "TransformerModel":
        return {
            "max_fights": model.max_fights,
            "d_model": model.input_proj.out_features,
            "num_layers": len(model.transformer.layers),
            "dropout": model.classifier[2].p,
        }
    return {"dropout": model.dropout.p}

def compute_feature_normalization(
    features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-feature means and standard deviations from a feature tensor.
    """
    means = features.mean(dim=0)
    stds = features.std(dim=0)
    stds[stds == 0] = 1.0
    return means, stds


def normalize_features(
    features: torch.Tensor,
    means: torch.Tensor,
    stds: torch.Tensor,
) -> torch.Tensor:
    """
    Apply precomputed normalization stats to a feature tensor.
    """
    return (features - means) / stds


def normalize_sequences(
    fighter1: torch.Tensor,
    fighter2: torch.Tensor,
    fighter1_mask: torch.Tensor,
    fighter2_mask: torch.Tensor,
    train_indices: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if train_indices is not None:
        fighter1_source = fighter1[train_indices]
        fighter2_source = fighter2[train_indices]
        mask1_source = fighter1_mask[train_indices]
        mask2_source = fighter2_mask[train_indices]
        combined = torch.cat([fighter1_source, fighter2_source], dim=0)
        combined_mask = torch.cat([mask1_source, mask2_source], dim=0)
    else:
        combined = torch.cat([fighter1, fighter2], dim=0)
        combined_mask = torch.cat([fighter1_mask, fighter2_mask], dim=0)
    valid_fights = combined[combined_mask.bool()]
    means = valid_fights.mean(dim=0)
    stds = valid_fights.std(dim=0)
    stds[stds == 0] = 1.0
    fighter1 = (fighter1 - means) / stds
    fighter2 = (fighter2 - means) / stds
    return fighter1, fighter2, means, stds

def optimize_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """
    Find a scalar temperature that minimizes validation NLL on held-out logits.
    """
    log_temperature = torch.zeros(1, device=logits.device, requires_grad=True)
    optimizer = optim.LBFGS([log_temperature], lr=0.1, max_iter=50)
    criterion = nn.CrossEntropyLoss()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp()
        loss = criterion(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return log_temperature.exp().item()

def save_artifacts(
    model: nn.Module,
    model_path: Union[str, Path],
    means: torch.Tensor,
    stds: torch.Tensor,
    normalization_path: Union[str, Path],
    temperature: float | None = None,
    matchup_means: torch.Tensor | None = None,
    matchup_stds: torch.Tensor | None = None,
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
    config = extract_model_config(model)
    torch.save(model.state_dict(), model_path)
    artifacts: dict[str, Any] = {"means": means, "stds": stds, "config": config}
    if temperature is not None:
        artifacts["temperature"] = temperature
    if matchup_means is not None and matchup_stds is not None:
        artifacts["matchup_means"] = matchup_means
        artifacts["matchup_stds"] = matchup_stds
    torch.save(artifacts, normalization_path)
    if VERBOSE: tqdm.write(f"Saved checkpoint to {model_path.parent}/")
