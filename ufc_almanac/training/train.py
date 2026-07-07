import argparse
from pathlib import Path
import time
from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import tqdm

from ufc_almanac.data import Data
from ufc_almanac.globals import (
    MAX_FIGHTS,
    STANDARD_TRAINING_DATA_PATH,
    TRANSFORMER_STANDARD_TRAINING_DATA_PATH,
)
from ufc_almanac.helpers import get_device, resolve_checkpoint_paths, resolve_model
from ufc_almanac.models import MODELS
from ufc_almanac.training.dataset import FightSequenceDataset
from ufc_almanac.training.utils import (
    collect_validation_logits,
    load_training_data,
    normalize_sequences,
    optimize_temperature,
    save_artifacts,
    temporal_train_val_split,
)


def evaluate(
    model: nn.Module,
    device: torch.device,
    data_loader: DataLoader,
    criterion: nn.Module,
    is_transformer: bool = False,
) -> tuple[float, float]:
    """
    Evaluates the performance of a model on a data loader.

    Args:
        model: torch.nn.Module
            The model to evaluate.
        device: torch.device
            The device to use for evaluation.
        data_loader: torch.utils.data.DataLoader
            The data loader to use for evaluation.
        criterion: torch.nn.Module
            The criterion to use for evaluation.

    Returns:
        tuple[float, float]
            The total loss and accuracy.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

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
                )
                total_loss += criterion(logits, labels).item() * labels.size(0)
                predictions = logits.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        else:
            for batch_features, batch_labels in data_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)

                logits = model(batch_features)
                total_loss += criterion(logits, batch_labels).item() * batch_labels.size(0)
                predictions = logits.argmax(dim=1)
                correct += (predictions == batch_labels).sum().item()
                total += batch_labels.size(0)

    return total_loss / total, (correct / total) * 100

def _training_run_description(
    base_desc: str,
    run_number: int | None,
    total_runs: int | None,
) -> str:
    if run_number is not None and total_runs is not None:
        return f"{base_desc} (run {run_number}/{total_runs})"
    return base_desc

def train_ff(
    training_data: dict[str, torch.Tensor],
    model: nn.Module,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    val_fraction: float,
    weight_decay: float,
    dropout: float,
    model_path: Path | None = None,
    save_checkpoint: bool = True,
    run_number: int | None = None,
    total_runs: int | None = None,
) -> dict[str, Any]:
    """
    Train the model using the training data and cross-entropy loss.

    Args:
        training_data: dict[str, torch.Tensor]
            The training data to use.
        model_class: nn.Module
            The class of the model to train.
        num_epochs: int
            The number of epochs to train for.
        batch_size: int
            The batch size to use.
        learning_rate: float
            The learning rate to use.
        val_fraction: float
            The fraction of the most recent samples (by fight date) held out for validation.
        weight_decay: float
            L2 regularization strength passed to the Adam optimizer.
        dropout: float
            Dropout probability applied within the model.
    """
    device = get_device()
    tqdm.write(f"Using device: {device}")

    features = training_data["features"]
    labels = training_data["labels"]

    means = features.mean(dim=0)
    stds = features.std(dim=0)
    stds[stds == 0] = 1.0
    features = (features - means) / stds

    dataset = TensorDataset(features, labels)
    train_indices, val_indices = temporal_train_val_split(
        len(dataset),
        val_fraction,
        training_data.get("fight_dates"),
    )
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = model(dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    resolved_model_path, resolved_normalization_path = resolve_checkpoint_paths(
        model.__class__,
        model_path=model_path,
    )

    start_time = time.time()
    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    best_model_state: dict[str, torch.Tensor] | None = None
    saved_during_training = False
    save_after_epoch = num_epochs / 3
    epoch_bar = tqdm(
        range(num_epochs),
        desc=_training_run_description("Training", run_number, total_runs),
        unit="epoch",
    )
    for epoch, _ in enumerate(epoch_bar):
        model.train()
        train_loss = 0.0

        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss, val_accuracy = evaluate(
            model, device, val_loader, criterion, is_transformer=False
        )
        if val_loss < best_val_loss:
            best_model_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
        if save_checkpoint and (epoch + 1) > save_after_epoch and val_loss < best_val_loss:
            save_artifacts(
                model,
                resolved_model_path,
                means,
                stds,
                resolved_normalization_path,
            )
            saved_during_training = True
        best_val_loss = min(best_val_loss, val_loss)
        best_val_accuracy = max(best_val_accuracy, val_accuracy)
        epoch_bar.set_postfix(
            train_loss=f"{train_loss / len(train_loader):.4f}",
            val_loss=f"{val_loss:.4f}",
            val_acc=f"{val_accuracy:.2f}%",
        )

    tqdm.write(f"Finished training in {round(time.time() - start_time, 1)} seconds")
    tqdm.write(
        f"Best val loss: {best_val_loss:.4f}, best val accuracy: {best_val_accuracy:.2f}%"
    )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    if save_checkpoint and not saved_during_training:
        save_artifacts(
            model,
            resolved_model_path,
            means,
            stds,
            resolved_normalization_path,
        )

    return {
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy,
        "model": model,
        "means": means,
        "stds": stds,
        "temperature": None,
        "model_path": resolved_model_path,
        "normalization_path": resolved_normalization_path,
    }

def train_transformer(
    training_data: dict[str, torch.Tensor],
    model: nn.Module,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    val_fraction: float,
    weight_decay: float,
    dropout: float,
    d_model: int,
    num_layers: int,
    model_path: Path | None = None,
    optimize_temp: bool = False,
    save_checkpoint: bool = True,
    run_number: int | None = None,
    total_runs: int | None = None,
) -> dict[str, Any]:
    device = get_device()
    tqdm.write(f"Using device: {device}")

    fighter1, fighter2, means, stds = normalize_sequences(
        training_data["fighter1"],
        training_data["fighter2"],
        training_data["fighter1_mask"],
        training_data["fighter2_mask"],
    )
    dataset = FightSequenceDataset(
        {
            "fighter1": fighter1,
            "fighter2": fighter2,
            "fighter1_mask": training_data["fighter1_mask"],
            "fighter2_mask": training_data["fighter2_mask"],
            "fighter1_days_before": training_data["fighter1_days_before"],
            "fighter2_days_before": training_data["fighter2_days_before"],
            "fighter1_days_gap": training_data["fighter1_days_gap"],
            "fighter2_days_gap": training_data["fighter2_days_gap"],
            "labels": training_data["labels"],
        }
    )

    train_indices, val_indices = temporal_train_val_split(
        len(dataset),
        val_fraction,
        training_data.get("fight_dates"),
    )
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = model(
        max_fights=int(training_data["max_fights"]),
        d_model=d_model,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    resolved_model_path, resolved_normalization_path = resolve_checkpoint_paths(
        model.__class__,
        model_path=model_path,
    )

    start_time = time.time()
    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    best_model_state: dict[str, torch.Tensor] | None = None
    saved_during_training = False
    save_after_epoch = num_epochs / 3
    epoch_bar = tqdm(
        range(num_epochs),
        desc=_training_run_description("Training transformer", run_number, total_runs),
        unit="epoch",
    )
    for epoch, _ in enumerate(epoch_bar):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            (
                fighter1,
                fighter2,
                mask1,
                mask2,
                days_before1,
                days_before2,
                days_gap1,
                days_gap2,
                labels,
            ) = [tensor.to(device) for tensor in batch]
            optimizer.zero_grad()
            logits = model(
                fighter1,
                fighter2,
                mask1,
                mask2,
                days_before1,
                days_before2,
                days_gap1,
                days_gap2,
            )
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss, val_accuracy = evaluate(
            model, device, val_loader, criterion, is_transformer=True
        )
        if val_loss < best_val_loss:
            best_model_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
        if save_checkpoint and (epoch + 1) > save_after_epoch and val_loss < best_val_loss:
            save_artifacts(
                model,
                resolved_model_path,
                means,
                stds,
                resolved_normalization_path,
            )
            saved_during_training = True
        best_val_loss = min(best_val_loss, val_loss)
        best_val_accuracy = max(best_val_accuracy, val_accuracy)
        epoch_bar.set_postfix(
            train_loss=f"{train_loss / len(train_loader):.4f}",
            val_loss=f"{val_loss:.4f}",
            val_acc=f"{val_accuracy:.2f}%",
        )

    tqdm.write(f"Finished training in {round(time.time() - start_time, 1)} seconds")
    tqdm.write(
        f"Best val loss: {best_val_loss:.4f}, best val accuracy: {best_val_accuracy:.2f}%"
    )

    temperature = None
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    if optimize_temp:
        val_logits, val_labels = collect_validation_logits(
            model,
            device,
            val_loader,
            is_transformer=True,
        )
        temperature = optimize_temperature(val_logits, val_labels)
        val_nll = nn.CrossEntropyLoss()(val_logits / temperature, val_labels).item()
        tqdm.write(
            f"Optimized temperature: {temperature:.4f} (val NLL: {val_nll:.4f})"
        )

    if save_checkpoint and (optimize_temp or not saved_during_training):
        save_artifacts(
            model,
            resolved_model_path,
            means,
            stds,
            resolved_normalization_path,
            temperature=temperature,
        )

    return {
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy,
        "model": model,
        "means": means,
        "stds": stds,
        "temperature": temperature,
        "model_path": resolved_model_path,
        "normalization_path": resolved_normalization_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a fight outcome model.")
    parser.add_argument(
        "--model",
        default="linear",
        choices=sorted(MODELS),
        help="model architecture to train (default: linear)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=400,
        help="number of training epochs (default: 400)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="training batch size (default: 256)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="optimizer learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="fraction of the most recent samples held out for validation (default: 0.2)",
    )
    parser.add_argument(
        "--rebuild-data",
        action="store_true",
        help="regenerate training data before training",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-3,
        help="L2 regularization strength for Adam (default: 1e-3)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="dropout probability (default: 0.5)",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=16,
        help="transformer hidden dimension (default: 16)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=12,
        help="number of transformer encoder layers (default: 12)",
    )
    parser.add_argument(
        "--max-fights",
        type=int,
        default=MAX_FIGHTS,
        help="past fights per fighter, i.e. sequence length (default: 8)",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="path to save trained model weights "
        "(default: artifacts/checkpoints/<ModelName>.pt)",
    )
    parser.add_argument(
        "--optimize-temp",
        action="store_true",
        help="optimize temperature scaling on the validation set after training "
        "(transformer only)",
    )
    parser.add_argument(
        "--restarts",
        type=int,
        default=1,
        help="number of independent training runs; when greater than 1, saves the run "
        "with the lowest validation loss (default: 1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.restarts < 1:
        raise ValueError("--restarts must be at least 1")

    transformer_model = "transformer" in args.model.lower()
    train_fn = train_transformer if transformer_model else train_ff
    data_path = Path(
        TRANSFORMER_STANDARD_TRAINING_DATA_PATH
        if transformer_model
        else STANDARD_TRAINING_DATA_PATH
    )

    if transformer_model:
        needs_rebuild = args.rebuild_data or not data_path.exists()
        if not needs_rebuild:
            existing_data = load_training_data(data_path)
            if int(existing_data["max_fights"]) != args.max_fights:
                needs_rebuild = True
            if "fight_dates" not in existing_data:
                needs_rebuild = True
            if "fighter1_days_before" not in existing_data:
                needs_rebuild = True
        if needs_rebuild:
            data_handler = Data()
            data_handler.create_transformer_training_data(max_fights=args.max_fights)
    else:
        needs_rebuild = args.rebuild_data or not data_path.exists()
        if not needs_rebuild:
            existing_data = load_training_data(data_path)
            if "fight_dates" not in existing_data:
                needs_rebuild = True
            if "fighter1_days_before" not in existing_data:
                needs_rebuild = True
        if needs_rebuild:
            data_handler = Data()
            data_handler.create_standard_training_data()

    training_data = load_training_data(data_path)
    train_kwargs = {
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "val_fraction": args.val_fraction,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "model_path": args.path,
    }
    if transformer_model:
        train_kwargs["d_model"] = args.d_model
        train_kwargs["num_layers"] = args.num_layers
        train_kwargs["optimize_temp"] = args.optimize_temp

    model_class = resolve_model(args.model, MODELS)
    save_checkpoint = args.restarts == 1
    run_results: list[tuple[int, dict[str, Any]]] = []

    for run_number in range(1, args.restarts + 1):
        if args.restarts > 1:
            tqdm.write(f"Training run {run_number}/{args.restarts}")
        result = train_fn(
            training_data,
            model_class,
            save_checkpoint=save_checkpoint,
            run_number=run_number if args.restarts > 1 else None,
            total_runs=args.restarts if args.restarts > 1 else None,
            **train_kwargs,
        )
        run_results.append((run_number, result))

    if args.restarts > 1:
        best_run_number, best_result = min(
            run_results,
            key=lambda item: item[1]["best_val_loss"],
        )
        tqdm.write("Restart summary:")
        for run_number, result in run_results:
            marker = "*" if run_number == best_run_number else " "
            tqdm.write(
                f"{marker} Run {run_number}: val loss {result['best_val_loss']:.4f}, "
                f"val accuracy {result['best_val_accuracy']:.2f}%"
            )
        tqdm.write(
            f"Saving run {best_run_number} with lowest val loss "
            f"({best_result['best_val_loss']:.4f})"
        )
        save_artifacts(
            best_result["model"],
            best_result["model_path"],
            best_result["means"],
            best_result["stds"],
            best_result["normalization_path"],
            temperature=best_result["temperature"],
        )


if __name__ == "__main__":
    main()
