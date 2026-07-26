import argparse
import datetime
import html
from pathlib import Path

from ufc_almanac.data import Data
from ufc_almanac.globals import CORE_TRANSFORMER_MODEL_PATH, OUTCOME_LABELS, VERBOSE
from ufc_almanac.inference import FightPredictor
from ufc_almanac.models import TransformerModel
from ufc_almanac.scraping import scrape_next_event


README_SECTION_START = "## Next UFC Event Predictions\n\n"
README_SECTION_END = "\n\nThe model used for"
TABLE_BORDER_COLOR = "#d0d7de"

WIN_METHOD_COLUMNS = [
    ("Overall", ("Win",)),
    ("KO", ("Win - KO/TKO",)),
    ("Sub", ("Win - Submission",)),
    (
        "Dec",
        (
            "Win - Unanimous Decision",
            "Win - Split Decision",
            "Win - Majority Decision",
        ),
    ),
]
LOSS_METHOD_COLUMNS = [
    ("Overall", ("Loss",)),
    ("KO", ("Loss - KO/TKO",)),
    ("Sub", ("Loss - Submission",)),
    (
        "Dec",
        (
            "Loss - Unanimous Decision",
            "Loss - Split Decision",
            "Loss - Majority Decision",
        ),
    ),
]


def table_cell_style(*, section_start: bool = False) -> str:
    styles = ["text-align: center;"]
    if section_start:
        styles.append(f"border-left: 1px solid {TABLE_BORDER_COLOR};")
    return f' style="{" ".join(styles)}"'


def overall_column_indices() -> set[int]:
    win_count = len(WIN_METHOD_COLUMNS)
    return {1, 1 + win_count}


def format_html_cell_content(content: str, *, bold: bool = False) -> str:
    escaped = html.escape(content)
    if bold:
        return f"<strong>{escaped}</strong>"
    return escaped


def format_html_header_cell(
    label: str,
    *,
    section_start: bool = False,
    bold: bool = False,
) -> str:
    return (
        f"<th{table_cell_style(section_start=section_start)}>"
        f"{format_html_cell_content(label, bold=bold)}</th>"
    )


def format_html_data_cell(
    content: str,
    *,
    section_start: bool = False,
    bold: bool = False,
) -> str:
    return (
        f"<td{table_cell_style(section_start=section_start)}>"
        f"{format_html_cell_content(content, bold=bold)}</td>"
    )


def section_start_indices() -> set[int]:
    win_count = len(WIN_METHOD_COLUMNS)
    loss_count = len(LOSS_METHOD_COLUMNS)
    return {1, 1 + win_count, 1 + win_count + loss_count}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict outcomes for the next UFC event."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path(CORE_TRANSFORMER_MODEL_PATH),
        help="path to trained model weights "
        f"(default: {CORE_TRANSFORMER_MODEL_PATH})",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=None,
        help="update predictions section in the README file",
    )
    return parser.parse_args()

def format_event_date(event_date: datetime.date) -> str:
    return f"{event_date.strftime('%B')} {event_date.day}, {event_date.year}"

def format_probability(probability: float | None) -> str:
    if probability is None:
        return "—"
    return f"{probability * 100:.1f}%"

def prediction_value(
    prediction: dict[str, float],
    key: str,
) -> float | None:
    value = prediction.get(key)
    if value is None:
        return None
    return float(value)

def prediction_column_value(
    prediction: dict[str, float],
    keys: tuple[str, ...],
) -> float | None:
    values = [prediction_value(prediction, key) for key in keys]
    if all(value is None for value in values):
        return None
    return sum(value or 0.0 for value in values)

def format_prediction_row(
    fighter1: str,
    fighter2: str,
    prediction: dict[str, float],
) -> list[str]:
    fight_label = f"{fighter1} vs {fighter2}"
    cells = [fight_label]
    for _, keys in WIN_METHOD_COLUMNS + LOSS_METHOD_COLUMNS:
        cells.append(format_probability(prediction_column_value(prediction, keys)))
    cells.append(format_probability(prediction_value(prediction, OUTCOME_LABELS[2])))
    return cells

def format_predictions_table(
    fights: list[tuple[str, str]],
    predictions: list[dict[str, float]],
    *,
    use_html: bool = False,
) -> str:
    if use_html:
        return _format_predictions_table_html(fights, predictions)
    return _format_predictions_table_markdown(fights, predictions)

def _format_predictions_table_markdown(
    fights: list[tuple[str, str]],
    predictions: list[dict[str, float]],
) -> str:
    win_headers = [
        f"Win (**{label}**)" if label == "Overall" else f"Win ({label})"
        for label, _ in WIN_METHOD_COLUMNS
    ]
    loss_headers = [
        f"Loss (**{label}**)" if label == "Overall" else f"Loss ({label})"
        for label, _ in LOSS_METHOD_COLUMNS
    ]
    headers = ["Fight", *win_headers, *loss_headers, "Draw"]
    separator = [":---:" if index > 0 else "---" for index in range(len(headers))]
    overall_indices = overall_column_indices()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for (fighter1, fighter2), prediction in zip(fights, predictions):
        row = format_prediction_row(fighter1, fighter2, prediction)
        formatted_row = [
            f"**{cell}**" if index in overall_indices else cell
            for index, cell in enumerate(row)
        ]
        lines.append("| " + " | ".join(formatted_row) + " |")
    return "\n".join(lines) + "\n"

def _format_predictions_table_html(
    fights: list[tuple[str, str]],
    predictions: list[dict[str, float]],
) -> str:
    win_count = len(WIN_METHOD_COLUMNS)
    loss_count = len(LOSS_METHOD_COLUMNS)
    section_starts = section_start_indices()
    overall_indices = overall_column_indices()
    win_subheaders = "".join(
        format_html_header_cell(
            label,
            section_start=index == 0,
            bold=label == "Overall",
        )
        for index, (label, _) in enumerate(WIN_METHOD_COLUMNS)
    )
    loss_subheaders = "".join(
        format_html_header_cell(
            label,
            section_start=index == 0,
            bold=label == "Overall",
        )
        for index, (label, _) in enumerate(LOSS_METHOD_COLUMNS)
    )
    rows = []
    for (fighter1, fighter2), prediction in zip(fights, predictions):
        cells = format_prediction_row(fighter1, fighter2, prediction)
        row_cells = "".join(
            format_html_data_cell(
                cell,
                section_start=index in section_starts,
                bold=index in overall_indices,
            )
            for index, cell in enumerate(cells)
        )
        rows.append(f"<tr>{row_cells}</tr>")

    return (
        f'<table style="margin: 0 auto; text-align: center; '
        f'border-collapse: collapse; border: 1px solid {TABLE_BORDER_COLOR};">\n'
        "<thead>\n"
        "<tr>\n"
        f'<th rowspan="2"{table_cell_style()}>Fight</th>\n'
        f'<th colspan="{win_count}"{table_cell_style(section_start=True)}>Win</th>\n'
        f'<th colspan="{loss_count}"{table_cell_style(section_start=True)}>Loss</th>\n'
        f'<th rowspan="2"{table_cell_style(section_start=True)}>Draw</th>\n'
        "</tr>\n"
        "<tr>\n"
        f"{win_subheaders}\n"
        f"{loss_subheaders}\n"
        "</tr>\n"
        "</thead>\n"
        "<tbody>\n"
        f"{''.join(rows)}\n"
        "</tbody>\n"
        "</table>\n"
    )

def format_predictions_section(
    event_date: str,
    fights: list[tuple[str, str]],
    predictions: list[dict[str, float]],
) -> str:
    table = format_predictions_table(fights, predictions, use_html=True)
    return f'Event date: {event_date}\n\n<div align="center">\n\n{table}\n</div>\n'

def update_readme(readme_path: Path, section_content: str) -> None:
    text = readme_path.read_text(encoding="utf-8")
    start_index = text.find(README_SECTION_START)
    if start_index == -1:
        raise ValueError(f"Could not find section header in {readme_path}")

    content_start = start_index + len(README_SECTION_START)
    end_index = text.find(README_SECTION_END, content_start)
    if end_index == -1:
        raise ValueError(f"Could not find section end marker in {readme_path}")

    readme_path.write_text(
        text[:content_start] + section_content + text[end_index:],
        encoding="utf-8",
    )

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
    display_date = format_event_date(date)
    section = format_predictions_section(display_date, fights, predictions)

    if args.readme is not None:
        update_readme(args.readme, section)

    print(f"Event date: {display_date}")
    print(format_predictions_table(fights, predictions))


if __name__ == "__main__":
    main()
