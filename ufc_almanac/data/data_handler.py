import pandas
from pathlib import Path
import torch
from tqdm import tqdm
from typing import Optional

from ufc_almanac.data.utils import (
    days_since_fight_date,
    filter_fighter_rows,
    load_csv,
    load_training_data,
    opposite_label,
    pad_fight_sequence,
    pad_temporal_sequence,
    parse_date_sort_key,
    per_minute_stats,
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

        self.standard_training_data = load_training_data(standard_training_path)
        self.transformer_training_data = load_training_data(transformer_training_path)

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
        Find the average stats of a fighter for the n most recent fights they had prior to a given date
        """
        days_since_fight = days_since_fight_date(date)
        fighter_info = self._lookup_fighter_profile(name)
        past_fights = self._get_sorted_past_fights(
            name, days_since_fight, min_fights, date=date
        )

        height = fighter_info["Height"]
        reach = fighter_info["Reach"]
        age = fighter_info["Age"]
        years_since = days_since_fight // 365

        fighter_useful_data = [
            height,
            reach,
            age - years_since,
        ]

        time = 0
        knockdown = 0
        knockdown_taken = 0
        sig_strikes_landed = 0
        sig_strikes_attempted = 0
        sig_strikes_absorbed = 0
        strikes_landed = 0
        strikes_attempted = 0
        strikes_absorbed = 0
        takedowns = 0
        takedown_attempts = 0
        got_takendown = 0
        submission_attempts = 0
        clinch_strikes = 0
        clinch_strikes_taken = 0
        ground_strikes = 0
        ground_strikes_taken = 0

        for row in past_fights[:min_fights]:
            time += int(row["Time"])
            knockdown += int(row["Knockdowns"])
            knockdown_taken += int(row["Knockdowns Against"])
            sig_strikes_landed += int(row["Sig Strikes Landed"])
            sig_strikes_attempted += int(row["Sig Strikes Attempted"])
            sig_strikes_absorbed += int(row["Sig Strikes Absorbed"])
            strikes_landed += int(row["Strikes Landed"])
            strikes_attempted += int(row["Strikes Attempted"])
            strikes_absorbed += int(row["Strikes Absorbed"])
            takedowns += int(row["Takedowns"])
            takedown_attempts += int(row["Takedown Attempts"])
            got_takendown += int(row["Got Taken Down"])
            submission_attempts += int(row["Submission Attempts"])
            clinch_strikes += int(row["Clinch Strikes"])
            clinch_strikes_taken += int(row["Clinch Strikes Taken"])
            ground_strikes += int(row["Ground Strikes"])
            ground_strikes_taken += int(row["Ground Strikes Taken"])

        # calculates the stats for the fighter, averaged over the total time they have spent in fights (per minute)
        knockdowns_pm = round((knockdown / (time / 60)), 4)
        gets_knockeddown_pm = round((knockdown_taken / (time / 60)), 4)
        sig_strikes_landed_pm = round((sig_strikes_landed / (time / 60)), 4)
        sig_strikes_attempted_pm = round((sig_strikes_attempted / (time / 60)), 4)
        sig_strikes_absorbed_pm = round((sig_strikes_absorbed / (time / 60)), 4)
        strikes_landed_pm = round((strikes_landed / (time / 60)), 4)
        strikes_attempted_pm = round((strikes_attempted / (time / 60)), 4)
        strikes_absorbed_pm = round((strikes_absorbed / (time / 60)), 4)
        strike_accuracy = round((strikes_landed / strikes_attempted), 4)
        takedowns_pm = round((takedowns / (time / 60)), 4)
        takedown_attempts_pm = round((takedown_attempts / (time / 60)), 4)
        gets_takendown_pm = round((got_takendown / (time / 60)), 4)
        submission_attempts_pm = round((submission_attempts / (time / 60)), 4)
        clinch_strikes_pm = round((clinch_strikes / (time / 60)), 4)
        clinch_strikes_taken_pm = round((clinch_strikes_taken / (time / 60)), 4)
        ground_strikes_pm = round((ground_strikes / (time / 60)), 4)
        ground_strikes_taken_pm = round((ground_strikes_taken / (time / 60)), 4)

        # adds all of the averaged stats to a list
        fighter_useful_data.append(knockdowns_pm)
        fighter_useful_data.append(gets_knockeddown_pm)
        fighter_useful_data.append(sig_strikes_landed_pm)
        fighter_useful_data.append(sig_strikes_attempted_pm)
        fighter_useful_data.append(sig_strikes_absorbed_pm)
        fighter_useful_data.append(strikes_landed_pm)
        fighter_useful_data.append(strikes_attempted_pm)
        fighter_useful_data.append(strikes_absorbed_pm)
        fighter_useful_data.append(strike_accuracy)
        fighter_useful_data.append(takedowns_pm)
        fighter_useful_data.append(takedown_attempts_pm)
        fighter_useful_data.append(gets_takendown_pm)
        fighter_useful_data.append(submission_attempts_pm)
        fighter_useful_data.append(clinch_strikes_pm)
        fighter_useful_data.append(clinch_strikes_taken_pm)
        fighter_useful_data.append(ground_strikes_pm)
        fighter_useful_data.append(ground_strikes_taken_pm)

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

        height = fighter_info["Height"]
        reach = fighter_info["Reach"]
        age = fighter_info["Age"]

        sequence = []
        days_before = []
        days_gap = []
        previous_days_since = None
        for row in past_fights[:max_fights]:
            row_days_since = days_since_fight_date(str(row["Date"]))
            years_since = (row_days_since - days_since_fight) // 365
            fight_features = [
                height,
                reach,
                age - years_since,
                *per_minute_stats(row),
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

            for (
                seq1,
                seq2,
                before1,
                before2,
                gap1,
                gap2,
                sample_label,
            ) in (
                (sequence1, sequence2, days_before1, days_before2, days_gap1, days_gap2, label),
                (sequence2, sequence1, days_before2, days_before1, days_gap2, days_gap1, opp_label),
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

                features.append(fighter1_useful_data + fighter2_useful_data)
                labels.append(label)
                fight_dates.append(date_key)

                # reversed sample to help the model generalise/reduce overfitting
                features.append(fighter2_useful_data + fighter1_useful_data)
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
