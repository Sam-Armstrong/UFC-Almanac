import datetime
import math
import pandas
import torch
from typing import Union

from ufc_almanac.globals import (
    MATCHUP_DAYS_SINCE_LAST_FIGHT_SIZE,
    MATCHUP_FIGHTER_PROFILE_FEATURE_SIZE,
    MATCHUP_STATIC_FEATURE_SIZE,
    METHOD_RECORD_FEATURE_SIZE,
    RECENCY_HALF_LIFE_DAYS,
)


KO_TKO_METHODS = {"KO/TKO", "TKO - Doctor's Stoppage"}
SUBMISSION_METHODS = {"Submission"}
DECISION_METHODS = {
    "Decision - Unanimous",
    "Decision - Split",
    "Decision - Majority",
}


def build_matchup_features(
    fighter1_profile: pandas.Series,
    fighter2_profile: pandas.Series,
    days_since_fight: int,
    days_before1: list[float] | None = None,
    days_before2: list[float] | None = None,
    fighter1_method_record: list[float] | None = None,
    fighter2_method_record: list[float] | None = None,
) -> list[float]:
    """
    Build matchup features for a head-to-head prediction.
    """
    days_since_last_fight1 = float(days_before1[0]) if days_before1 else 0.0
    days_since_last_fight2 = float(days_before2[0]) if days_before2 else 0.0

    features = [
        *fighter_matchup_profile_features(fighter1_profile, days_since_fight),
        *fighter_matchup_profile_features(fighter2_profile, days_since_fight),
        days_since_last_fight1,
        days_since_last_fight2,
    ]
    if fighter1_method_record is not None and fighter2_method_record is not None:
        features.extend(fighter1_method_record)
        features.extend(fighter2_method_record)
    return features

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

def fight_method_category(method: str) -> str | None:
    """
    Map a fight result method to ko_tko, submission, decision, or None.
    """
    normalized_method = str(method).strip()
    if normalized_method in KO_TKO_METHODS:
        return "ko_tko"
    if normalized_method in SUBMISSION_METHODS:
        return "submission"
    if normalized_method in DECISION_METHODS:
        return "decision"
    return None

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

def fighter_matchup_profile_features(
    fighter_profile: pandas.Series,
    days_since_fight: int,
) -> list[float]:
    """
    Build raw reach, height, age, and stance features for a fighter at a fight date.
    """
    years_since = days_since_fight // 365
    age = float(fighter_profile["Age"] - years_since)
    return [
        float(fighter_profile["Reach"]),
        float(fighter_profile["Height"]),
        age,
        *stance_binary_features(float(fighter_profile["Stance"])),
    ]

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

def mirror_matchup_features(matchup: list[float]) -> list[float]:
    """
    Flip matchup features when swapping fighter 1 and fighter 2.
    """
    profile_size = MATCHUP_FIGHTER_PROFILE_FEATURE_SIZE
    fighter1_profile = matchup[:profile_size]
    fighter2_profile = matchup[profile_size : 2 * profile_size]
    days_start = 2 * profile_size
    days1 = matchup[days_start]
    days2 = matchup[days_start + MATCHUP_DAYS_SINCE_LAST_FIGHT_SIZE - 1]
    method_start = MATCHUP_STATIC_FEATURE_SIZE
    fighter1_method_record = matchup[
        method_start : method_start + METHOD_RECORD_FEATURE_SIZE
    ]
    fighter2_method_record = matchup[
        method_start + METHOD_RECORD_FEATURE_SIZE : method_start
        + 2 * METHOD_RECORD_FEATURE_SIZE
    ]
    return [
        *fighter2_profile,
        *fighter1_profile,
        days2,
        days1,
        *fighter2_method_record,
        *fighter1_method_record,
    ]

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

def stance_binary_features(stance: float) -> list[float]:
    """
    Encode stance as orthodox / southpaw / switch one-hot features.
    """
    stance_value = int(stance)
    return [
        1.0 if stance_value == 1 else 0.0,
        1.0 if stance_value == 2 else 0.0,
        1.0 if stance_value == 3 else 0.0,
    ]
