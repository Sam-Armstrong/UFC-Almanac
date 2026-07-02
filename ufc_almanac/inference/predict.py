import argparse
from datetime import date
from pathlib import Path
import torch
from typing import Optional

from ufc_almanac.data import Data
from ufc_almanac.helpers import get_device, resolve_model
from ufc_almanac.models import MODELS, TransformerModel
from ufc_almanac.globals import (
    INPUT_SIZE,
    LABEL_COLUMNS,
    MAX_FIGHTS,
    MIN_FIGHTS,
    NUM_CLASSES,
    TRANSFORMER_FEATURE_SIZE,
    TRANSFORMER_STANDARD_TRAINING_DATA_PATH,
)


class FightPredictor:
    def __init__(self, model: torch.nn.Module) -> None:
        self.device = get_device()
        self.is_transformer = model is TransformerModel
        self.max_fights = self._resolve_max_fights() if self.is_transformer else MAX_FIGHTS
        feature_size = TRANSFORMER_FEATURE_SIZE if self.is_transformer else INPUT_SIZE

        if self.is_transformer:
            self.model = model(max_fights=self.max_fights).to(self.device)
        else:
            self.model = model().to(self.device)

        model_name = self.model.__class__.__name__
        self.model_path = Path("saved") / f"{model_name}.pt"
        self.normalization_path = Path("saved") / f"{model_name}_normalization.pt"
        self.means = torch.zeros(feature_size)
        self.stds = torch.ones(feature_size)
        self._load_artifacts()

    def _resolve_max_fights(self) -> int:
        training_path = Path(TRANSFORMER_STANDARD_TRAINING_DATA_PATH)
        if training_path.exists():
            training_data = torch.load(training_path, weights_only=True)
            return int(training_data["max_fights"])
        return MAX_FIGHTS

    def _load_artifacts(self) -> None:
        if not self.model_path.exists() or not self.normalization_path.exists():
            return

        self.model.load_state_dict(
            torch.load(self.model_path, map_location=self.device, weights_only=True)
        )
        normalization = torch.load(
            self.normalization_path,
            map_location=self.device,
            weights_only=True,
        )
        self.means = normalization["means"]
        self.stds = normalization["stds"]

    def _normalize(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.means.to(self.device)) / self.stds.to(self.device)

    def _prepare_features(
        self,
        fighter1_stats: list,
        fighter2_stats: list,
    ) -> torch.Tensor:
        features = torch.tensor(
            fighter1_stats + fighter2_stats,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        return self._normalize(features)

    def _prepare_transformer_features(
        self,
        fighter1_sequence: list[list[float]],
        fighter2_sequence: list[list[float]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        padded1, mask1 = Data._pad_fight_sequence(fighter1_sequence, self.max_fights)
        padded2, mask2 = Data._pad_fight_sequence(fighter2_sequence, self.max_fights)

        fighter1 = torch.tensor(
            padded1,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        fighter2 = torch.tensor(
            padded2,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        fighter1_mask = torch.tensor(
            mask1,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        fighter2_mask = torch.tensor(
            mask2,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        fighter1 = self._normalize(fighter1)
        fighter2 = self._normalize(fighter2)
        return fighter1, fighter2, fighter1_mask, fighter2_mask

    def _probabilities_from_logits(self, logits: torch.Tensor) -> dict[str, float]:
        probabilities = torch.softmax(logits, dim=-1).squeeze(0)
        return {
            LABEL_COLUMNS[index]: probabilities[index].item()
            for index in range(NUM_CLASSES)
        }

    def predict(
        self,
        fighter1_stats: list,
        fighter2_stats: list,
    ) -> dict[str, float]:
        """
        Return win / loss / draw probabilities for fighter 1.
        """
        self.model.eval()
        features = self._prepare_features(fighter1_stats, fighter2_stats)

        with torch.no_grad():
            logits = self.model(features)

        return self._probabilities_from_logits(logits)

    def predict_sequences(
        self,
        fighter1_sequence: list[list[float]],
        fighter2_sequence: list[list[float]],
    ) -> dict[str, float]:
        """
        Return win / loss / draw probabilities for fighter 1 using fight sequences.
        """
        self.model.eval()
        fighter1, fighter2, fighter1_mask, fighter2_mask = (
            self._prepare_transformer_features(fighter1_sequence, fighter2_sequence)
        )

        with torch.no_grad():
            logits = self.model(fighter1, fighter2, fighter1_mask, fighter2_mask)

        return self._probabilities_from_logits(logits)

    def predict_fighters(
        self,
        data: Data,
        fighter1: str,
        fighter2: str,
        date: str,
        min_fights: Optional[int] = None,
    ) -> dict[str, float]:
        """
        Build feature vectors from fighter names and return outcome probabilities.
        """
        if min_fights is None:
            min_fights = MIN_FIGHTS

        if self.is_transformer:
            fighter1_sequence = data.get_fight_sequence(
                fighter1,
                date,
                min_fights=min_fights,
                max_fights=self.max_fights,
            )
            fighter2_sequence = data.get_fight_sequence(
                fighter2,
                date,
                min_fights=min_fights,
                max_fights=self.max_fights,
            )
            return self.predict_sequences(fighter1_sequence, fighter2_sequence)

        fighter1_stats = data.find_fighter_stats(fighter1, date, min_fights=min_fights)
        fighter2_stats = data.find_fighter_stats(fighter2, date, min_fights=min_fights)
        return self.predict(fighter1_stats, fighter2_stats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict fight outcomes.")
    parser.add_argument(
        "--model",
        default="linear",
        choices=sorted(MODELS),
        help="model architecture to load (default: linear)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = FightPredictor(resolve_model(args.model, MODELS))
    data = Data()

    break_works = [
        "",
        "exit",
        "quit",
        "q",
    ]

    while True:
        fighter1 = input("Enter the name of the first fighter: ")
        if fighter1.lower() in break_works:
            break

        fighter2 = input("Enter the name of the second fighter: ")
        if fighter2.lower() in break_works:
            break

        result = predictor.predict_fighters(
            data,
            fighter1,
            fighter2,
            str(date.today()),
        )
        percentages = {label: value * 100 for label, value in result.items()}
        print(
            f"{fighter1} Win: {percentages['Win']:.2f}%, "
            f"{fighter1} Loss: {percentages['Loss']:.2f}%, "
            f"Draw: {percentages['Draw']:.2f}%"
        )


if __name__ == "__main__":
    main()
