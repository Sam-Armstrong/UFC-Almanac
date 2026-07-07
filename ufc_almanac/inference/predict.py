import argparse
from datetime import date
from pathlib import Path
import torch
from typing import Optional

from ufc_almanac.inference.utils import (
    infer_transformer_config,
    load_model_state_dict,
    load_normalization_artifacts,
)
from ufc_almanac.data import Data, pad_fight_sequence, pad_temporal_sequence
from ufc_almanac.helpers import get_device, resolve_checkpoint_paths, resolve_model
from ufc_almanac.models import MODELS
from ufc_almanac.models.transformer import apply_temperature
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
    def __init__(
        self,
        model: type[torch.nn.Module],
        model_path: Optional[Path] = None,
    ) -> None:
        self.device = get_device()
        self.is_transformer = model.__name__ == "TransformerModel"
        feature_size = TRANSFORMER_FEATURE_SIZE if self.is_transformer else INPUT_SIZE

        self.model_path, self.normalization_path = resolve_checkpoint_paths(
            model,
            model_path=model_path,
        )
        normalization = self._load_normalization()
        state_dict = self._load_state_dict()
        model_kwargs = self._resolve_model_kwargs(state_dict, normalization.get("config", {}))
        self.max_fights = model_kwargs.get("max_fights", MAX_FIGHTS)
        self.model = model(**model_kwargs).to(self.device)

        if state_dict is not None:
            self.model.load_state_dict(state_dict)

        self.means = normalization.get("means", torch.zeros(feature_size))
        self.stds = normalization.get("stds", torch.ones(feature_size))
        self.temperature = float(normalization.get("temperature", 1.0))

    def _load_state_dict(self) -> Optional[dict[str, torch.Tensor]]:
        return load_model_state_dict(self.model_path, self.device)

    def _load_normalization(self) -> dict:
        return load_normalization_artifacts(self.normalization_path, self.device)

    def _resolve_model_kwargs(
        self,
        state_dict: Optional[dict[str, torch.Tensor]],
        saved_config: dict,
    ) -> dict:
        if self.is_transformer:
            config = infer_transformer_config(state_dict) if state_dict else {}
            config.update(saved_config)
            if "max_fights" not in config:
                config["max_fights"] = self._resolve_max_fights()
            return {
                "max_fights": int(config["max_fights"]),
                "d_model": int(config.get("d_model", 64)),
                "num_layers": int(config.get("num_layers", 2)),
                "dropout": float(config.get("dropout", 0.1)),
            }

        return {"dropout": float(saved_config.get("dropout", 0.0))}

    def _resolve_max_fights(self) -> int:
        training_path = Path(TRANSFORMER_STANDARD_TRAINING_DATA_PATH)
        if training_path.exists():
            training_data = torch.load(training_path, weights_only=True)
            return int(training_data["max_fights"])
        return MAX_FIGHTS

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
        fighter1_days_before: list[float],
        fighter2_days_before: list[float],
        fighter1_days_gap: list[float],
        fighter2_days_gap: list[float],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        padded1, mask1 = pad_fight_sequence(fighter1_sequence, self.max_fights)
        padded2, mask2 = pad_fight_sequence(fighter2_sequence, self.max_fights)
        padded_days_before1 = pad_temporal_sequence(fighter1_days_before, self.max_fights)
        padded_days_before2 = pad_temporal_sequence(fighter2_days_before, self.max_fights)
        padded_days_gap1 = pad_temporal_sequence(fighter1_days_gap, self.max_fights)
        padded_days_gap2 = pad_temporal_sequence(fighter2_days_gap, self.max_fights)

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
        fighter1_days_before = torch.tensor(
            padded_days_before1,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        fighter2_days_before = torch.tensor(
            padded_days_before2,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        fighter1_days_gap = torch.tensor(
            padded_days_gap1,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        fighter2_days_gap = torch.tensor(
            padded_days_gap2,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        fighter1 = self._normalize(fighter1)
        fighter2 = self._normalize(fighter2)
        return (
            fighter1,
            fighter2,
            fighter1_mask,
            fighter2_mask,
            fighter1_days_before,
            fighter2_days_before,
            fighter1_days_gap,
            fighter2_days_gap,
        )

    def _probabilities_from_logits(
        self,
        logits: torch.Tensor,
        sig_figs: int = 4,
    ) -> dict[str, float]:
        probabilities = torch.round(
            torch.softmax(
                apply_temperature(logits, self.temperature),
                dim=-1,
            ).squeeze(0),
            decimals=sig_figs,
        )
        return {
            LABEL_COLUMNS[index]: probabilities[index].item()
            for index in range(NUM_CLASSES)
        }

    def predict(
        self,
        fighter1_stats: list,
        fighter2_stats: list,
        sig_figs: int = 4,
    ) -> dict[str, float]:
        """
        Return win / loss / draw probabilities for fighter 1.
        """
        self.model.eval()
        features = self._prepare_features(fighter1_stats, fighter2_stats)

        with torch.no_grad():
            logits = self.model(features)

        return self._probabilities_from_logits(logits, sig_figs=sig_figs)

    def predict_sequences(
        self,
        fighter1_sequence: list[list[float]],
        fighter2_sequence: list[list[float]],
        fighter1_days_before: list[float],
        fighter2_days_before: list[float],
        fighter1_days_gap: list[float],
        fighter2_days_gap: list[float],
        sig_figs: int = 4,
    ) -> dict[str, float]:
        """
        Return win / loss / draw probabilities for fighter 1 using fight sequences.
        """
        self.model.eval()
        (
            fighter1,
            fighter2,
            fighter1_mask,
            fighter2_mask,
            fighter1_days_before_tensor,
            fighter2_days_before_tensor,
            fighter1_days_gap_tensor,
            fighter2_days_gap_tensor,
        ) = self._prepare_transformer_features(
            fighter1_sequence,
            fighter2_sequence,
            fighter1_days_before,
            fighter2_days_before,
            fighter1_days_gap,
            fighter2_days_gap,
        )

        with torch.no_grad():
            logits = self.model(
                fighter1,
                fighter2,
                fighter1_mask,
                fighter2_mask,
                fighter1_days_before_tensor,
                fighter2_days_before_tensor,
                fighter1_days_gap_tensor,
                fighter2_days_gap_tensor,
            )

        return self._probabilities_from_logits(logits, sig_figs=sig_figs)

    def predict_fighters(
        self,
        data: Data,
        fighter1: str,
        fighter2: str,
        date: str,
        min_fights: Optional[int] = None,
        sig_figs: int = 4,
    ) -> dict[str, float]:
        """
        Build feature vectors from fighter names and return outcome probabilities.

        Args:
            data: Data object containing fighter statistics
            fighter1: Name of the first fighter
            fighter2: Name of the second fighter
            date: Date of the fight in format YYYY-MM-DD
            min_fights: Minimum number of fights to consider for the prediction
            sig_figs: Number of significant figures to round the probabilities to

        Returns:
            Dictionary containing win / loss / draw probabilities for fighter 1
        """
        if min_fights is None:
            min_fights = MIN_FIGHTS

        if self.is_transformer:
            fighter1_sequence, fighter1_days_before, fighter1_days_gap = (
                data.get_fight_sequence(
                    fighter1,
                    date,
                    min_fights=min_fights,
                    max_fights=self.max_fights,
                )
            )
            fighter2_sequence, fighter2_days_before, fighter2_days_gap = (
                data.get_fight_sequence(
                    fighter2,
                    date,
                    min_fights=min_fights,
                    max_fights=self.max_fights,
                )
            )
            return self.predict_sequences(
                fighter1_sequence,
                fighter2_sequence,
                fighter1_days_before,
                fighter2_days_before,
                fighter1_days_gap,
                fighter2_days_gap,
                sig_figs=sig_figs,
            )

        fighter1_stats = data.find_fighter_stats(fighter1, date, min_fights=min_fights)
        fighter2_stats = data.find_fighter_stats(fighter2, date, min_fights=min_fights)
        return self.predict(fighter1_stats, fighter2_stats, sig_figs=sig_figs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict fight outcomes.")
    parser.add_argument(
        "--model",
        default="linear",
        choices=sorted(MODELS),
        help="model architecture to load (default: linear)",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="path to trained model weights "
        "(default: artifacts/checkpoints/<ModelName>.pt)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = FightPredictor(
        resolve_model(args.model, MODELS),
        model_path=args.path,
    )
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
