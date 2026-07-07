import datetime
import pandas
import torch
from typing import Union


def calculate_days_since(day: str, month: str, year: str) -> int:
    """
    Calculates the days between a given date and the current date
    """
    a = datetime.date(int(year), int(month), int(day))
    b = datetime.date.today()
    days_since = b - a
    days_since = str(days_since)

    if len(days_since.split(" ")) > 1:
        days_since = int(days_since.split(" ")[0])
    else:
        days_since = 0

    return days_since

def days_since_fight_date(date: str) -> int:
    if "/" in date:
        day, month, year = date.split("/")
    else:
        year, month, day = date.split("-")
    return calculate_days_since(day, month, year)

def load_csv(path: str) -> Union[pandas.DataFrame, None]:
    """
    Load a CSV file, dropping any legacy index column
    """
    try:
        dataframe = pandas.read_csv(path)
        if "Unnamed: 0" in dataframe.columns:
            dataframe = dataframe.drop(columns=["Unnamed: 0"])
        return dataframe
    except FileNotFoundError:
        return None

def load_training_data(path: str) -> Union[torch.Tensor, None]:
    """
    Load a training data file, dropping any legacy index column
    """
    if path.exists():
        return torch.load(path, weights_only=True)
    return None

def opposite_label(result: int) -> int:
    if result == 3:
        return 2
    return 1 if result == 1 else 0

def pad_fight_sequence(
    sequence: list[list[float]],
    max_fights: int,
) -> tuple[list[list[float]], list[float]]:
    feature_size = len(sequence[0])
    padded = [[0.0] * feature_size for _ in range(max_fights)]
    mask = [0.0] * max_fights
    for index, fight in enumerate(sequence[:max_fights]):
        padded[index] = fight
        mask[index] = 1.0
    return padded, mask

def pad_temporal_sequence(
    values: list[float],
    max_fights: int,
) -> list[float]:
    padded = [0.0] * max_fights
    for index, value in enumerate(values[:max_fights]):
        padded[index] = value
    return padded

def parse_date_sort_key(date: str) -> int:
    """
    Return a YYYYMMDD integer for chronological sorting of fight dates.
    """
    if "/" in date:
        day, month, year = date.split("/")
    else:
        year, month, day = date.split("-")
    return int(year) * 10000 + int(month) * 100 + int(day)

def per_minute_stats(row: pandas.Series) -> list[float]:
    time = max(int(row["Time"]), 1)
    minutes = time / 60
    knockdown = int(row["Knockdowns"])
    knockdown_taken = int(row["Knockdowns Against"])
    sig_strikes_landed = int(row["Sig Strikes Landed"])
    sig_strikes_attempted = int(row["Sig Strikes Attempted"])
    sig_strikes_absorbed = int(row["Sig Strikes Absorbed"])
    strikes_landed = int(row["Strikes Landed"])
    strikes_attempted = int(row["Strikes Attempted"])
    strikes_absorbed = int(row["Strikes Absorbed"])
    takedowns = int(row["Takedowns"])
    takedown_attempts = int(row["Takedown Attempts"])
    got_takendown = int(row["Got Taken Down"])
    submission_attempts = int(row["Submission Attempts"])
    clinch_strikes = int(row["Clinch Strikes"])
    clinch_strikes_taken = int(row["Clinch Strikes Taken"])
    ground_strikes = int(row["Ground Strikes"])
    ground_strikes_taken = int(row["Ground Strikes Taken"])

    return [
        round(knockdown / minutes, 4),
        round(knockdown_taken / minutes, 4),
        round(sig_strikes_landed / minutes, 4),
        round(sig_strikes_attempted / minutes, 4),
        round(sig_strikes_absorbed / minutes, 4),
        round(strikes_landed / minutes, 4),
        round(strikes_attempted / minutes, 4),
        round(strikes_absorbed / minutes, 4),
        round(strikes_landed / max(strikes_attempted, 1), 4),
        round(takedowns / minutes, 4),
        round(takedown_attempts / minutes, 4),
        round(got_takendown / minutes, 4),
        round(submission_attempts / minutes, 4),
        round(clinch_strikes / minutes, 4),
        round(clinch_strikes_taken / minutes, 4),
        round(ground_strikes / minutes, 4),
        round(ground_strikes_taken / minutes, 4),
    ]
