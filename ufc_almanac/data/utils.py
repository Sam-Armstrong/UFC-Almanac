import datetime
import math
import pandas
import torch
from typing import Union

from ufc_almanac.globals import (
    RECENCY_HALF_LIFE_DAYS,
    WEIGHT_CLASS_MISMATCH_THRESHOLD,
)


def build_matchup_features(
    fighter1_profile: pandas.Series,
    fighter2_profile: pandas.Series,
    days_since_fight: int,
    days_before1: list[float] | None = None,
    days_before2: list[float] | None = None,
) -> list[float]:
    """
    Build matchup-relative features for a head-to-head prediction.
    """
    height1 = float(fighter1_profile["Height"])
    height2 = float(fighter2_profile["Height"])
    reach1 = float(fighter1_profile["Reach"])
    reach2 = float(fighter2_profile["Reach"])
    weight1 = float(fighter1_profile["Weight"])
    weight2 = float(fighter2_profile["Weight"])
    stance1 = float(fighter1_profile["Stance"])
    stance2 = float(fighter2_profile["Stance"])
    years_since = days_since_fight // 365
    age1 = float(fighter1_profile["Age"] - years_since)
    age2 = float(fighter2_profile["Age"] - years_since)
    weight_diff = abs(weight1 - weight2)
    days_since_last_fight1 = float(days_before1[0]) if days_before1 else 0.0
    days_since_last_fight2 = float(days_before2[0]) if days_before2 else 0.0

    return [
        reach1 - reach2,
        height1 - height2,
        age1 - age2,
        weight1 - weight2,
        1.0 if weight_diff > WEIGHT_CLASS_MISMATCH_THRESHOLD else 0.0,
        1.0 if stance1 != stance2 else 0.0,
        days_since_last_fight1,
        days_since_last_fight2,
    ]

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

def fight_outcome_for_fighter(fight_row: pandas.Series, fighter_name: str) -> float:
    """
    Encode a fighter's past fight outcome as win=1.0, loss=0.0, draw=0.5.
    """
    result = int(fight_row["Result"])
    if result == 3:
        return 0.5
    fighter1 = str(fight_row["Fighter 1"]).strip()
    if normalize_fighter_name(fighter_name) == normalize_fighter_name(fighter1):
        return 1.0 if result == 1 else 0.0
    return 1.0 if result == 2 else 0.0

def fighter_age_at_fight(current_age: float, fight_days_since: int, matchup_days_since: int) -> float:
    """
    Estimate a fighter's age at a past fight relative to a future matchup date.
    """
    years_since = (fight_days_since - matchup_days_since) // 365
    return float(current_age - years_since)

def filter_fighter_rows(dataframe: pandas.DataFrame, name: str) -> pandas.DataFrame:
    """
    Return rows whose fighter name matches exactly after normalization.
    """
    normalized_name = normalize_fighter_name(name)
    return dataframe[
        dataframe["Name"].map(normalize_fighter_name) == normalized_name
    ]

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

def normalize_fighter_name(name: str) -> str:
    """
    Normalize a fighter name for exact matching.
    """
    return str(name).strip().casefold()

def opponent_name_for_fighter(fight_row: pandas.Series, fighter_name: str) -> str:
    """
    Return the opponent name from a fight result row.
    """
    fighter1 = str(fight_row["Fighter 1"]).strip()
    fighter2 = str(fight_row["Fighter 2"]).strip()
    if normalize_fighter_name(fighter_name) == normalize_fighter_name(fighter1):
        return fighter2
    return fighter1

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

def recency_weight(days_before: float, half_life_days: float = RECENCY_HALF_LIFE_DAYS) -> float:
    """
    Exponential recency weight for a past fight.
    """
    return math.exp(-days_before / half_life_days)
