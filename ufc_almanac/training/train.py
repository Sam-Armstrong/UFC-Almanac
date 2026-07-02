import argparse
from pathlib import Path
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from ufc_almanac.data import Data
from ufc_almanac.globals import (
    CHECKPOINTS_DIR,
    STANDARD_TRAINING_DATA_PATH,
    TRANSFORMER_STANDARD_TRAINING_DATA_PATH,
)
from ufc_almanac.helpers import get_device, resolve_model
from ufc_almanac.models import MODELS
from ufc_almanac.training.dataset import FightSequenceDataset
from ufc_almanac.training.utils import (
    load_training_data,
    normalize_sequences,
    save_artifacts,
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
                fighter1, fighter2, mask1, mask2, labels = [
                    tensor.to(device) for tensor in batch
                ]
                logits = model(fighter1, fighter2, mask1, mask2)
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

def train_ff(
    training_data: dict[str, torch.Tensor],
    model: nn.Module,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    val_fraction: float,
    weight_decay: float,
    dropout: float,
) -> None:
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
            The fraction of the training data to use for validation.
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
    val_size = max(1, int(len(dataset) * val_fraction))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = model(dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    start_time = time.time()
    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    epoch_bar = tqdm(range(num_epochs), desc="Training", unit="epoch")
    for _ in epoch_bar:
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
    model_name = model.__class__.__name__
    checkpoint_dir = Path(CHECKPOINTS_DIR)
    save_artifacts(
        model,
        checkpoint_dir / f"{model_name}.pt",
        means,
        stds,
        checkpoint_dir / f"{model_name}_normalization.pt",
    )

def train_transformer(
    training_data: dict[str, torch.Tensor],
    model: nn.Module,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    val_fraction: float,
    weight_decay: float,
    dropout: float,
) -> None:
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
            "labels": training_data["labels"],
        }
    )

    val_size = max(1, int(len(dataset) * val_fraction))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = model(max_fights=training_data["max_fights"], dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    start_time = time.time()
    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    epoch_bar = tqdm(range(num_epochs), desc="Training transformer", unit="epoch")
    for _ in epoch_bar:
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            fighter1, fighter2, mask1, mask2, labels = [
                tensor.to(device) for tensor in batch
            ]
            optimizer.zero_grad()
            logits = model(fighter1, fighter2, mask1, mask2)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss, val_accuracy = evaluate(
            model, device, val_loader, criterion, is_transformer=True
        )
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
    model_name = model.__class__.__name__
    checkpoint_dir = Path(CHECKPOINTS_DIR)
    save_artifacts(
        model,
        checkpoint_dir / f"{model_name}.pt",
        means,
        stds,
        checkpoint_dir / f"{model_name}_normalization.pt",
    )


# Main #


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
        default=50,
        help="number of training epochs (default: 50)",
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
        default=0.1,
        help="fraction of data held out for validation (default: 0.1)",
    )
    parser.add_argument(
        "--rebuild-data",
        action="store_true",
        help="regenerate training data before training",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="L2 regularization strength for Adam (default: 1e-4)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="dropout probability (default: 0.2)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    transformer_model = "transformer" in args.model.lower()
    train_fn = train_transformer if transformer_model else train_ff
    data_path = Path(
        TRANSFORMER_STANDARD_TRAINING_DATA_PATH
        if transformer_model
        else STANDARD_TRAINING_DATA_PATH
    )

    if args.rebuild_data or not data_path.exists():
        data_handler = Data()
        if transformer_model:
            data_handler.create_transformer_training_data()
        else:
            data_handler.create_standard_training_data()

    training_data = load_training_data(data_path)
    train_fn(
        training_data,
        resolve_model(args.model, MODELS),
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_fraction=args.val_fraction,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
    )


if __name__ == "__main__":
    main()
