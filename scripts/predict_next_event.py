import argparse
from pathlib import Path

from ufc_almanac.data import Data
from ufc_almanac.globals import VERBOSE
from ufc_almanac.inference import FightPredictor
from ufc_almanac.models import TransformerModel
from ufc_almanac.scraping import scrape_next_event


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict outcomes for the next UFC event."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("artifacts/core/transformer_model.pt"),
        help="path to trained model weights "
        "(default: artifacts/core/transformer_model.pt)",
    )
    return parser.parse_args()


def format_predictions_table(
    fights: list[tuple[str, str]],
    predictions: list[dict[str, float]],
) -> str:
    lines = [
        "| Fight | Win % | Loss % | Draw % |",
        "| --- | --- | --- | --- |",
    ]
    for (fighter1, fighter2), result in zip(fights, predictions):
        win_pct = result["Win"] * 100
        loss_pct = result["Loss"] * 100
        draw_pct = result["Draw"] * 100
        fight_label = f"{fighter1} vs {fighter2}"
        lines.append(
            f"| {fight_label} | {win_pct:.1f}% | {loss_pct:.1f}% | {draw_pct:.1f}% |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    upcoming_event = scrape_next_event()
    date = upcoming_event.date
    fights = upcoming_event.fights

    data = Data()
    predictor = FightPredictor(TransformerModel, model_path=args.path)

    predictions = []
    skipped_fights = []
    for fighter1, fighter2 in fights:
        try:
            prediction = predictor.predict_fighters(data, fighter1, fighter2, str(date), sig_figs=3)
            predictions.append(prediction)
        except Exception as e:
            if VERBOSE: print(f"Skipping {fighter1} vs {fighter2}: {e}")
            skipped_fights.append((fighter1, fighter2))
            continue

    fights = [fight for fight in fights if fight not in skipped_fights]
    print(f"Event date: {date}")
    print(format_predictions_table(fights, predictions))


if __name__ == "__main__":
    main()
