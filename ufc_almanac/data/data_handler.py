import pandas
from pathlib import Path
import torch
from tqdm import tqdm
from typing import Optional

from ufc_almanac.data.utils import (
    build_matchup_features,
    days_since_fight_date,
    fight_outcome_for_fighter,
    filter_fighter_rows,
    fighter_age_at_fight,
    load_csv,
    load_training_data,
    normalize_fighter_name,
    opponent_name_for_fighter,
    opposite_label,
    pad_fight_sequence,
    pad_temporal_sequence,
    parse_date_sort_key,
    per_minute_stats,
    recency_weight,
)
from ufc_almanac.exceptions import (
    MinFightsException,
    MissingDataException,
    MissingFighterDataException,
)
from ufc_almanac.globals import (
    FIGHTER_DATA_CSV,
    MAX_FIGHTS,
    MIN_FIGHTS,
    RESULTS_CSV,
    STATS_CSV,
    STANDARD_TRAINING_DATA_PATH,
    TRANSFORMER_STANDARD_TRAINING_DATA_PATH,
    VERBOSE,
)


class Data:
    def __init__(self):
        standard_training_path = Path(STANDARD_TRAINING_DATA_PATH)
        transformer_training_path = Path(TRANSFORMER_STANDARD_TRAINING_DATA_PATH)

        self.fight_results = load_csv(RESULTS_CSV)
        self.fight_stats = load_csv(STATS_CSV)
        self.fighter_data = load_csv(FIGHTER_DATA_CSV)
        self.standard_training_data = load_training_data(standard_training_path)
        self.transformer_training_data = load_training_data(transformer_training_path)
        self._fight_result_index: dict[tuple[str, str], pandas.Series] = {}
        self._fighter_history: dict[str, list[tuple[int, float]]] = {}
        self._indexes_built = False

    def _ensure_indexes(self) -> None:
        if self._indexes_built:
            return

        for _, row in self.fight_results.iterrows():
            result = int(row["Result"])
            if result == 4:
                continue

            date = str(row["Date"])
            row_days_since = days_since_fight_date(date)
            outcome_fighter1 = fight_outcome_for_fighter(row, str(row["Fighter 1"]).strip())
            outcome_fighter2 = fight_outcome_for_fighter(row, str(row["Fighter 2"]).strip())

            for fighter_name, outcome in (
                (str(row["Fighter 1"]).strip(), outcome_fighter1),
                (str(row["Fighter 2"]).strip(), outcome_fighter2),
            ):
                normalized_name = normalize_fighter_name(fighter_name)
                self._fight_result_index[(normalized_name, date)] = row
                self._fighter_history.setdefault(normalized_name, []).append(
                    (row_days_since, outcome)
                )

        for outcomes in self._fighter_history.values():
            outcomes.sort(key=lambda item: item[0])

        self._indexes_built = True

    def _lookup_fight_result(self, name: str, date: str) -> pandas.Series | None:
        self._ensure_indexes()
        return self._fight_result_index.get((normalize_fighter_name(name), date))

    def _fighter_win_rate_before(self, name: str, days_since_cutoff: int) -> float:
        self._ensure_indexes()
        history = self._fighter_history.get(normalize_fighter_name(name), [])
        wins = 0.0
        losses = 0.0
        for row_days_since, outcome in history:
            if row_days_since <= days_since_cutoff:
                continue
            if outcome == 1.0:
                wins += 1.0
            elif outcome == 0.0:
                losses += 1.0
        return wins / max(1.0, wins + losses)

    def _safe_lookup_fighter_profile(self, name: str) -> pandas.Series | None:
        try:
            return self._lookup_fighter_profile(name)
        except MissingFighterDataException:
            return None

    def _opponent_context_for_past_fight(
        self,
        fighter_name: str,
        fight_date: str,
        days_since_fight: int,
    ) -> list[float]:
        fight_row = self._lookup_fight_result(fighter_name, fight_date)
        if fight_row is None:
            return [0.0] * 6

        opponent_name = opponent_name_for_fighter(fight_row, fighter_name)
        opponent_profile = self._safe_lookup_fighter_profile(opponent_name)
        fight_days_since = days_since_fight_date(fight_date)
        outcome = fight_outcome_for_fighter(fight_row, fighter_name)

        if opponent_profile is None:
            return [0.0, 0.0, 0.0, 0.0, 0.0, outcome]

        return [
            float(opponent_profile["Height"]),
            float(opponent_profile["Reach"]),
            fighter_age_at_fight(
                float(opponent_profile["Age"]),
                fight_days_since,
                days_since_fight,
            ),
            float(opponent_profile["Weight"]),
            self._fighter_win_rate_before(opponent_name, fight_days_since),
            outcome,
        ]

    def get_matchup_features(
        self,
        name1: str,
        name2: str,
        date: str,
        days_before1: list[float] | None = None,
        days_before2: list[float] | None = None,
    ) -> list[float]:
        """
        Build matchup-relative features for two fighters at a given date.
        """
        days_since_fight = days_since_fight_date(date)
        fighter1_profile = self._lookup_fighter_profile(name1)
        fighter2_profile = self._lookup_fighter_profile(name2)
        return build_matchup_features(
            fighter1_profile,
            fighter2_profile,
            days_since_fight,
            days_before1=days_before1,
            days_before2=days_before2,
        )

    def _lookup_fighter_profile(self, name: str) -> pandas.Series:
        """
        Return the fighter profile row for an exact name match.
        """
        matches = filter_fighter_rows(self.fighter_data, name)
        if len(matches) != 1:
            raise MissingFighterDataException(f"Missing data for fighter {name}")
        return matches.iloc[0]

    def _get_sorted_past_fights(
        self,
        name: str,
        days_since_fight: int,
        min_fights: int,
        date: Optional[str] = None,
    ) -> list[pandas.Series]:
        """
        Return a fighter's past fights prior to a given date, most-recent-first.
        """
        fighter_data = filter_fighter_rows(self.fight_stats, name)
        past_fights: list[tuple[int, pandas.Series]] = []
        for _, row in fighter_data.iterrows():
            row_days_since = days_since_fight_date(str(row["Date"]))
            if row_days_since > days_since_fight:
                past_fights.append((row_days_since, row))

        past_fights.sort(key=lambda item: item[0])
        if len(past_fights) < min_fights:
            date_suffix = f" at {date}" if date is not None else ""
            raise MinFightsException(
                f"Fighter {name} had fewer than {min_fights} fights{date_suffix}"
            )
        return [row for _, row in past_fights]

    def find_fighter_stats(self, name: str, date: str, min_fights: int = 3) -> list[float]:
        """
        Find recency-weighted average stats for a fighter's most recent prior fights.
        """
        days_since_fight = days_since_fight_date(date)
        fighter_info = self._lookup_fighter_profile(name)
        past_fights = self._get_sorted_past_fights(
            name, days_since_fight, min_fights, date=date
        )
        recent_fights = past_fights[:min_fights]

        height = float(fighter_info["Height"])
        reach = float(fighter_info["Reach"])
        weight = float(fighter_info["Weight"])
        stance = float(fighter_info["Stance"])
        years_since = days_since_fight // 365

        fighter_useful_data = [
            height,
            reach,
            float(fighter_info["Age"] - years_since),
            weight,
            stance,
        ]

        weighted_stats: list[float] = []
        total_weight = 0.0
        recent_win_rate_weight = 0.0
        opponent_win_rate_weight = 0.0
        days_since_last_fight = 0.0

        for index, row in enumerate(recent_fights):
            row_days_since = days_since_fight_date(str(row["Date"]))
            days_before = float(row_days_since - days_since_fight)
            weight_value = recency_weight(days_before)
            if index == 0:
                days_since_last_fight = days_before

            fight_row = self._lookup_fight_result(name, str(row["Date"]))
            if fight_row is not None:
                outcome = fight_outcome_for_fighter(fight_row, name)
                recent_win_rate_weight += outcome * weight_value
                opponent_name = opponent_name_for_fighter(fight_row, name)
                opponent_win_rate = self._fighter_win_rate_before(
                    opponent_name,
                    row_days_since,
                )
                opponent_win_rate_weight += opponent_win_rate * weight_value

            if not weighted_stats:
                weighted_stats = [0.0] * len(per_minute_stats(row))
            for stat_index, stat_value in enumerate(per_minute_stats(row)):
                weighted_stats[stat_index] += stat_value * weight_value
            total_weight += weight_value

        if total_weight == 0.0:
            raise MinFightsException(
                f"Fighter {name} had fewer than {min_fights} fights at {date}"
            )

        fighter_useful_data.extend(
            round(stat / total_weight, 4) for stat in weighted_stats
        )
        fighter_useful_data.append(round(recent_win_rate_weight / total_weight, 4))
        fighter_useful_data.append(round(opponent_win_rate_weight / total_weight, 4))
        fighter_useful_data.append(days_since_last_fight)

        return fighter_useful_data

    def get_fight_sequence(
        self,
        name: str,
        date: str,
        min_fights: int = MIN_FIGHTS,
        max_fights: int = MAX_FIGHTS,
    ) -> tuple[list[list[float]], list[float], list[float]]:
        """
        Return per-fight feature vectors for a fighter's past fights prior to a given date.
        Fights are ordered most-recent-first.

        Also returns days_before (days from each past fight to the given date) and
        days_gap (days between consecutive past fights, 0 for the most recent fight).
        """
        days_since_fight = days_since_fight_date(date)
        fighter_info = self._lookup_fighter_profile(name)
        past_fights = self._get_sorted_past_fights(
            name,
            days_since_fight,
            min_fights,
            date=date,
        )

        height = float(fighter_info["Height"])
        reach = float(fighter_info["Reach"])
        age = float(fighter_info["Age"])
        weight = float(fighter_info["Weight"])
        stance = float(fighter_info["Stance"])

        sequence = []
        days_before = []
        days_gap = []
        previous_days_since = None
        for row in past_fights[:max_fights]:
            fight_date = str(row["Date"])
            row_days_since = days_since_fight_date(fight_date)
            years_since = (row_days_since - days_since_fight) // 365
            fight_features = [
                height,
                reach,
                age - years_since,
                weight,
                stance,
                *per_minute_stats(row),
                *self._opponent_context_for_past_fight(
                    name,
                    fight_date,
                    days_since_fight,
                ),
            ]
            sequence.append(fight_features)
            days_before.append(float(row_days_since - days_since_fight))
            if previous_days_since is None:
                days_gap.append(0.0)
            else:
                days_gap.append(float(row_days_since - previous_days_since))
            previous_days_since = row_days_since

        return sequence, days_before, days_gap

    def create_transformer_training_data(
        self,
        min_fights: Optional[int] = None,
        max_fights: Optional[int] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Build padded fight-sequence training data for the transformer model.

        Each sample contains both fighters' past fights (most recent first),
        with labels for fighter 1's win / loss / draw outcome.
        """
        if any(
            [
                df is None or len(df) == 0
                for df in [self.fight_results, self.fight_stats, self.fighter_data]
            ]
        ):
            raise MissingDataException()

        if min_fights is None:
            min_fights = MIN_FIGHTS
        if max_fights is None:
            max_fights = MAX_FIGHTS
        if save_path is None:
            save_path = TRANSFORMER_STANDARD_TRAINING_DATA_PATH

        fighter1_sequences = []
        fighter2_sequences = []
        fighter1_masks = []
        fighter2_masks = []
        fighter1_days_before = []
        fighter2_days_before = []
        fighter1_days_gap = []
        fighter2_days_gap = []
        matchup_features = []
        labels = []
        fight_dates = []

        for _, row in tqdm(
            self.fight_results.iterrows(),
            total=len(self.fight_results),
            desc="Creating transformer training data",
            unit="fight",
        ):
            date = str(row["Date"])
            if date <= "01/01/2010":
                continue

            name1 = str(row["Fighter 1"]).strip()
            name2 = str(row["Fighter 2"]).strip()
            result = int(row["Result"])

            # skip no contest fights
            if result == 4: continue

            try:
                sequence1, days_before1, days_gap1 = self.get_fight_sequence(
                    name1, date, min_fights, max_fights
                )
                sequence2, days_before2, days_gap2 = self.get_fight_sequence(
                    name2, date, min_fights, max_fights
                )
            except Exception as e:
                if VERBOSE:
                    tqdm.write(f"Skipping fight: {e}")
                continue

            label = result - 1
            opp_label = opposite_label(result)
            date_key = parse_date_sort_key(date)
            matchup = self.get_matchup_features(
                name1,
                name2,
                date,
                days_before1=days_before1,
                days_before2=days_before2,
            )

            for (
                seq1,
                seq2,
                before1,
                before2,
                gap1,
                gap2,
                sample_label,
                sample_matchup,
            ) in (
                (
                    sequence1,
                    sequence2,
                    days_before1,
                    days_before2,
                    days_gap1,
                    days_gap2,
                    label,
                    matchup,
                ),
                (
                    sequence2,
                    sequence1,
                    days_before2,
                    days_before1,
                    days_gap2,
                    days_gap1,
                    opp_label,
                    [
                        -matchup[0],
                        -matchup[1],
                        -matchup[2],
                        -matchup[3],
                        matchup[4],
                        matchup[5],
                        matchup[7],
                        matchup[6],
                    ],
                ),
            ):
                padded1, mask1 = pad_fight_sequence(seq1, max_fights)
                padded2, mask2 = pad_fight_sequence(seq2, max_fights)
                fighter1_sequences.append(padded1)
                fighter2_sequences.append(padded2)
                fighter1_masks.append(mask1)
                fighter2_masks.append(mask2)
                fighter1_days_before.append(pad_temporal_sequence(before1, max_fights))
                fighter2_days_before.append(pad_temporal_sequence(before2, max_fights))
                fighter1_days_gap.append(pad_temporal_sequence(gap1, max_fights))
                fighter2_days_gap.append(pad_temporal_sequence(gap2, max_fights))
                matchup_features.append(sample_matchup)
                labels.append(sample_label)
                fight_dates.append(date_key)

        save_path_obj = Path(save_path) if isinstance(save_path, str) else save_path
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "fighter1": torch.tensor(fighter1_sequences, dtype=torch.float32),
                "fighter2": torch.tensor(fighter2_sequences, dtype=torch.float32),
                "fighter1_mask": torch.tensor(fighter1_masks, dtype=torch.float32),
                "fighter2_mask": torch.tensor(fighter2_masks, dtype=torch.float32),
                "fighter1_days_before": torch.tensor(
                    fighter1_days_before, dtype=torch.float32
                ),
                "fighter2_days_before": torch.tensor(
                    fighter2_days_before, dtype=torch.float32
                ),
                "fighter1_days_gap": torch.tensor(fighter1_days_gap, dtype=torch.float32),
                "fighter2_days_gap": torch.tensor(fighter2_days_gap, dtype=torch.float32),
                "matchup_features": torch.tensor(matchup_features, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.long),
                "fight_dates": torch.tensor(fight_dates, dtype=torch.long),
                "max_fights": max_fights,
            },
            save_path,
        )
        tqdm.write(f"Saved transformer training data to {save_path}")
        tqdm.write(f"Training samples: {len(labels)}")

    def create_standard_training_data(
        self,
        min_fights: Optional[int] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Creates a set of training data based upon the statistics of each fighter prior to a given fight,
        using the result of the fight as the training label

        Args:
            min_fights: int or None
                The minimum number of fights a fighter must have had to be considered for training.
                Default to MIN_FIGHTS from globals.py
            save_path: str or None
                The path to save the training data to.
                Default to STANDARD_TRAINING_DATA_PATH from globals.py
        """
        if any(
            [
                df is None or len(df) == 0
                for df in [self.fight_results, self.fight_stats, self.fighter_data]
            ]
        ):
            raise MissingDataException()

        if min_fights is None:
            min_fights = MIN_FIGHTS
        if save_path is None:
            save_path = STANDARD_TRAINING_DATA_PATH

        features = []
        labels = []
        fight_dates = []

        # loop through fight results and find the stats for each of the fighters from their n prior fights
        for _, row in tqdm(
            self.fight_results.iterrows(),
            total=len(self.fight_results),
            desc="Creating training data",
            unit="fight",
        ):
            date = str(row["Date"])
            name1 = str(row["Fighter 1"]).strip()
            name2 = str(row["Fighter 2"]).strip()
            result = int(row["Result"])

            if date > "01/01/2010" and result != 4:

                # finds the stats of the two fighters prior to the date of the given fight occuring
                try:
                    fighter1_useful_data = self.find_fighter_stats(
                        name1, date, min_fights
                    )
                    fighter2_useful_data = self.find_fighter_stats(
                        name2, date, min_fights
                    )
                except Exception as e:
                    if VERBOSE: tqdm.write(f"Skipping fight: {e}")
                    continue

                label = result - 1
                opp_label = opposite_label(result)
                date_key = parse_date_sort_key(date)
                matchup = self.get_matchup_features(
                    name1,
                    name2,
                    date,
                    days_before1=[fighter1_useful_data[-1]],
                    days_before2=[fighter2_useful_data[-1]],
                )

                features.append(fighter1_useful_data + fighter2_useful_data + matchup)
                labels.append(label)
                fight_dates.append(date_key)

                features.append(
                    fighter2_useful_data
                    + fighter1_useful_data
                    + [
                        -matchup[0],
                        -matchup[1],
                        -matchup[2],
                        -matchup[3],
                        matchup[4],
                        matchup[5],
                        matchup[7],
                        matchup[6],
                    ]
                )
                labels.append(opp_label)
                fight_dates.append(date_key)

        save_path_obj = Path(save_path) if isinstance(save_path, str) else save_path
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        self.training_data = {
            "features": torch.tensor(features, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "fight_dates": torch.tensor(fight_dates, dtype=torch.long),
        }
        torch.save(self.training_data, save_path)
        tqdm.write(f"Saved training data to {save_path}")
        tqdm.write(f"Training samples: {len(labels)}")


if __name__ == "__main__":
    data = Data()
    data.create_training_data()
    print(f"Training data length: {len(data.training_data['labels'])}")
