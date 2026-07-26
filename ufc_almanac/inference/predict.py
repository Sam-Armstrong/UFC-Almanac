import argparse
from datetime import date
from pathlib import Path
import torch
from typing import Optional

from ufc_almanac.data.utils import aggregate_outcome_probabilities
from ufc_almanac.inference.utils import (
    infer_num_classes,
    infer_transformer_config,
    load_model_state_dict,
    load_normalization_artifacts,
    validate_transformer_checkpoint,
)
from ufc_almanac.data import Data, pad_fight_sequence, pad_temporal_sequence
from ufc_almanac.helpers import get_device, resolve_checkpoint_paths, resolve_model
from ufc_almanac.models import MODELS
from ufc_almanac.models.transformer import apply_temperature
from ufc_almanac.globals import (
    CORE_TRANSFORMER_MODEL_PATH,
    INPUT_SIZE,
    LABEL_COLUMNS,
    MATCHUP_FEATURE_SIZE,
    MAX_FIGHTS,
    MIN_FIGHTS,
    OUTCOME_LABELS,
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
        if self.is_transformer and state_dict is not None:
            validate_transformer_checkpoint(state_dict)
        model_kwargs = self._resolve_model_kwargs(state_dict, normalization.get("config", {}))
        self.num_classes = model_kwargs.pop("num_classes")
        self.class_labels = self._resolve_class_labels(self.num_classes)
        self.max_fights = model_kwargs.get("max_fights", MAX_FIGHTS)
        self.model = model(**model_kwargs).to(self.device)

        if state_dict is not None:
            self.model.load_state_dict(state_dict)

        self.means = normalization.get("means", torch.zeros(feature_size))
        self.stds = normalization.get("stds", torch.ones(feature_size))
        self.matchup_means = normalization.get(
            "matchup_means",
            torch.zeros(MATCHUP_FEATURE_SIZE),
        )
        self.matchup_stds = normalization.get(
            "matchup_stds",
            torch.ones(MATCHUP_FEATURE_SIZE),
        )
        self.temperature = float(normalization.get("temperature", 1.0))

    def _load_state_dict(self) -> Optional[dict[str, torch.Tensor]]:
        return load_model_state_dict(self.model_path, self.device)

    def _load_normalization(self) -> dict:
        return load_normalization_artifacts(self.normalization_path, self.device)

    def _resolve_class_labels(self, num_classes: int) -> list[str]:
        if num_classes == len(LABEL_COLUMNS):
            return LABEL_COLUMNS
        if num_classes == len(OUTCOME_LABELS):
            return OUTCOME_LABELS
        raise ValueError(
            f"Unsupported number of model classes: {num_classes}. "
            f"Expected {len(LABEL_COLUMNS)} outcome-method classes or "
            f"{len(OUTCOME_LABELS)} legacy outcome classes."
        )

    def _resolve_model_kwargs(
        self,
        state_dict: Optional[dict[str, torch.Tensor]],
        saved_config: dict,
    ) -> dict:
        num_classes = infer_num_classes(state_dict) if state_dict else len(LABEL_COLUMNS)
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
                "num_classes": num_classes,
            }

        return {
            "dropout": float(saved_config.get("dropout", 0.0)),
            "num_classes": num_classes,
        }

    def _resolve_max_fights(self) -> int:
        training_path = Path(TRANSFORMER_STANDARD_TRAINING_DATA_PATH)
        if training_path.exists():
            training_data = torch.load(training_path, weights_only=True)
            return int(training_data["max_fights"])
        return MAX_FIGHTS

    def _normalize(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.means.to(self.device)) / self.stds.to(self.device)

    def _normalize_matchup(self, matchup_features: torch.Tensor) -> torch.Tensor:
        return (
            (matchup_features - self.matchup_means.to(self.device))
            / self.matchup_stds.to(self.device)
        )

    def _prepare_features(
        self,
        fighter1_stats: list,
        fighter2_stats: list,
        matchup_features: list,
    ) -> torch.Tensor:
        features = torch.tensor(
            fighter1_stats + fighter2_stats + matchup_features,
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
        matchup_features: list[float],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
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
        matchup = torch.tensor(
            matchup_features,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        matchup = self._normalize_matchup(matchup)
        return (
            fighter1,
            fighter2,
            fighter1_mask,
            fighter2_mask,
            fighter1_days_before,
            fighter2_days_before,
            fighter1_days_gap,
            fighter2_days_gap,
            matchup,
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
        class_probabilities = {
            self.class_labels[index]: probabilities[index].item()
            for index in range(self.num_classes)
        }
        if self.num_classes == len(LABEL_COLUMNS):
            return {
                **class_probabilities,
                **aggregate_outcome_probabilities(class_probabilities),
            }
        return class_probabilities

    def predict(
        self,
        fighter1_stats: list,
        fighter2_stats: list,
        matchup_features: list,
        sig_figs: int = 4,
    ) -> dict[str, float]:
        """
        Return outcome-method and aggregated win / loss / draw probabilities for fighter 1.
        """
        self.model.eval()
        features = self._prepare_features(
            fighter1_stats,
            fighter2_stats,
            matchup_features,
        )

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
        matchup_features: list[float],
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
            matchup_tensor,
        ) = self._prepare_transformer_features(
            fighter1_sequence,
            fighter2_sequence,
            fighter1_days_before,
            fighter2_days_before,
            fighter1_days_gap,
            fighter2_days_gap,
            matchup_features,
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
                matchup_tensor,
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
            Dictionary containing outcome-method probabilities and aggregated
            win / loss / draw probabilities for fighter 1
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
            matchup_features = data.get_matchup_features(
                fighter1,
                fighter2,
                date,
                days_before1=fighter1_days_before,
                days_before2=fighter2_days_before,
            )
            return self.predict_sequences(
                fighter1_sequence,
                fighter2_sequence,
                fighter1_days_before,
                fighter2_days_before,
                fighter1_days_gap,
                fighter2_days_gap,
                matchup_features,
                sig_figs=sig_figs,
            )

        fighter1_stats = data.find_fighter_stats(fighter1, date, min_fights=min_fights)
        fighter2_stats = data.find_fighter_stats(fighter2, date, min_fights=min_fights)
        matchup_features = data.get_matchup_features(
            fighter1,
            fighter2,
            date,
            days_before1=[fighter1_stats[-1]],
            days_before2=[fighter2_stats[-1]],
        )
        return self.predict(
            fighter1_stats,
            fighter2_stats,
            matchup_features,
            sig_figs=sig_figs,
        )


def resolve_default_model_path(model: type[torch.nn.Module]) -> Path | None:
    """
    Return the default inference checkpoint for models that ship a core artifact.
    """
    if model.__name__ == "TransformerModel":
        return Path(CORE_TRANSFORMER_MODEL_PATH)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict fight outcomes.")
    parser.add_argument(
        "--model",
        default="transformer",
        choices=sorted(MODELS),
        help="model architecture to load (default: transformer)",
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
    model = resolve_model(args.model, MODELS)
    model_path = args.path if args.path is not None else resolve_default_model_path(model)
    predictor = FightPredictor(
        model,
        model_path=model_path,
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
        outcome_percentages = {
            label: result[label] * 100 for label in OUTCOME_LABELS
        }
        print(
            f"{fighter1} Win: {outcome_percentages['Win']:.2f}%, "
            f"{fighter1} Loss: {outcome_percentages['Loss']:.2f}%, "
            f"Draw: {outcome_percentages['Draw']:.2f}%"
        )
        method_percentages = {
            label: result[label] * 100
            for label in predictor.class_labels
            if label != "Draw"
        }
        top_methods = sorted(
            method_percentages.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:3]
        method_summary = ", ".join(
            f"{label}: {value:.2f}%" for label, value in top_methods
        )
        print(f"Top methods: {method_summary}")


if __name__ == "__main__":
    main()
