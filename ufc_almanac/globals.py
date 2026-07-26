import os


FIGHTER_DATA_CSV = "data/FighterData.csv"
RESULTS_CSV = "data/FightResults.csv"
STATS_CSV = "data/FightStats.csv"
STANDARD_TRAINING_DATA_PATH = "data/StandardTrainingData.pt"
TRANSFORMER_STANDARD_TRAINING_DATA_PATH = "data/TransformerTrainingData.pt"
CHECKPOINTS_DIR = "artifacts/checkpoints"
CORE_TRANSFORMER_MODEL_PATH = "artifacts/core/transformer_model.pt"

BLOCKED_RESOURCE_TYPES = {"image", "media", "font"}
MIN_FIGHTS = 1
MAX_FIGHTS = 8
VERBOSE = bool(int(os.getenv("VERBOSE", 0)))
RECENCY_HALF_LIFE_DAYS = 365.0

STAT_COLUMNS = [
    "Knockdowns PM",
    "Gets Knocked Down PM",
    "Sig Strikes Landed PM",
    "Sig Strikes Attempted PM",
    "Sig Strikes Absorbed PM",
    "Strikes Landed PM",
    "Strikes Attempted PM",
    "Strikes Absorbed PM",
    "Strike Accuracy",
    "Takedowns PM",
    "Takedown Attempts PM",
    "Gets Taken Down PM",
    "Submission Attempts PM",
    "Clinch Strikes PM",
    "Clinch Strikes Taken PM",
    "Ground Strikes PM",
    "Ground Strikes Taken PM",
    "Fight Duration Minutes",
    "Sig Strike Accuracy",
    "Sig Strike Differential PM",
    "Takedown Accuracy",
    "Strike Defense",
]

FIGHT_METHOD_COLUMNS = [
    "Method KO/TKO",
    "Method Submission",
    "Method Decision",
]

FIGHTER_PROFILE_COLUMNS = [
    "Height",
    "Reach",
    "Age",
    "Weight",
    "Orthodox",
    "Southpaw",
    "Switch",
]

FIGHTER_FORM_COLUMNS = [
    "Recent Win Rate",
    "Avg Opponent Win Rate",
    "Avg Opponent Finish Rate",
    "Recent Method KO/TKO",
    "Recent Method Submission",
    "Recent Method Decision",
    "Days Since Last Fight",
]

FIGHTER_FEATURE_COLUMNS = (
    FIGHTER_PROFILE_COLUMNS
    + STAT_COLUMNS
    + FIGHTER_FORM_COLUMNS
)

METHOD_RECORD_COLUMNS = [
    "Wins by KO/TKO",
    "Wins by Submission",
    "Wins by Decision",
    "Losses by KO/TKO",
    "Losses by Submission",
    "Losses by Decision",
]
METHOD_RECORD_FEATURE_SIZE = len(METHOD_RECORD_COLUMNS)
MATCHUP_FIGHTER_PROFILE_COLUMNS = [
    "Reach",
    "Height",
    "Age",
    "Orthodox",
    "Southpaw",
    "Switch",
]
MATCHUP_FIGHTER_PROFILE_FEATURE_SIZE = len(MATCHUP_FIGHTER_PROFILE_COLUMNS)
MATCHUP_DAYS_SINCE_LAST_FIGHT_SIZE = 2
MATCHUP_STATIC_FEATURE_SIZE = (
    2 * MATCHUP_FIGHTER_PROFILE_FEATURE_SIZE + MATCHUP_DAYS_SINCE_LAST_FIGHT_SIZE
)

MATCHUP_FEATURE_COLUMNS = [
    *[f"{column} 1" for column in MATCHUP_FIGHTER_PROFILE_COLUMNS],
    *[f"{column} 2" for column in MATCHUP_FIGHTER_PROFILE_COLUMNS],
    "Days Since Last Fight 1",
    "Days Since Last Fight 2",
    *[f"{column} 1" for column in METHOD_RECORD_COLUMNS],
    *[f"{column} 2" for column in METHOD_RECORD_COLUMNS],
]

FEATURE_COLUMNS = (
    [f"{column} 1" for column in FIGHTER_FEATURE_COLUMNS]
    + [f"{column} 2" for column in FIGHTER_FEATURE_COLUMNS]
    + MATCHUP_FEATURE_COLUMNS
)

# Feature scaling policy:
# - standard: z-score using train-set mean/std
# - unit: already in [0, 1]; leave as-is (pre-normalized)
# - categorical: one-hot / binary / ordinal codes; leave as-is
UNIT_INTERVAL_FEATURE_NAMES = frozenset(
    {
        "Strike Accuracy",
        "Sig Strike Accuracy",
        "Takedown Accuracy",
        "Strike Defense",
        "Recent Win Rate",
        "Avg Opponent Win Rate",
        "Avg Opponent Finish Rate",
        "Recent Method KO/TKO",
        "Recent Method Submission",
        "Recent Method Decision",
        "Opponent Win Rate",
        "Opponent Finish Rate",
    }
)
CATEGORICAL_FEATURE_NAMES = frozenset(
    {
        "Orthodox",
        "Southpaw",
        "Switch",
        "Method KO/TKO",
        "Method Submission",
        "Method Decision",
        "Fight Outcome",
    }
)
IDENTITY_FEATURE_NAMES = UNIT_INTERVAL_FEATURE_NAMES | CATEGORICAL_FEATURE_NAMES


def _feature_base_name(column: str) -> str:
    if column.endswith(" 1") or column.endswith(" 2"):
        return column[:-2]
    return column


def identity_feature_indices(columns: list[str]) -> list[int]:
    """
    Return indices for features that should not be z-score standardized.
    """
    return [
        index
        for index, column in enumerate(columns)
        if _feature_base_name(column) in IDENTITY_FEATURE_NAMES
    ]


OUTCOME_METHOD_LABELS = [
    "Draw",
    "Win - KO/TKO",
    "Loss - KO/TKO",
    "Win - Submission",
    "Loss - Submission",
    "Win - Unanimous Decision",
    "Loss - Unanimous Decision",
    "Win - Split Decision",
    "Loss - Split Decision",
    "Win - Majority Decision",
    "Loss - Majority Decision",
]
OUTCOME_LABELS = ["Win", "Loss", "Draw"]
LABEL_COLUMNS = OUTCOME_METHOD_LABELS
INPUT_SIZE = len(FEATURE_COLUMNS)
NUM_CLASSES = len(LABEL_COLUMNS)
MATCHUP_FEATURE_SIZE = len(MATCHUP_FEATURE_COLUMNS)

TRANSFORMER_OPPONENT_COLUMNS = [
    "Opponent Height",
    "Opponent Reach",
    "Opponent Age",
    "Opponent Weight",
    "Opponent Win Rate",
    "Opponent Finish Rate",
    "Opponent Wins by KO/TKO",
    "Opponent Wins by Submission",
    "Opponent Wins by Decision",
    "Opponent Losses by KO/TKO",
    "Opponent Losses by Submission",
    "Opponent Losses by Decision",
    "Opponent Sig Strikes Landed PM",
    "Opponent Takedowns PM",
    "Opponent Submission Attempts PM",
    "Fight Outcome",
]

TRANSFORMER_FEATURE_COLUMNS = (
    FIGHTER_PROFILE_COLUMNS
    + STAT_COLUMNS
    + FIGHT_METHOD_COLUMNS
    + TRANSFORMER_OPPONENT_COLUMNS
)
TRANSFORMER_FEATURE_SIZE = len(TRANSFORMER_FEATURE_COLUMNS)
TRANSFORMER_STATIC_FEATURE_SIZE = len(FIGHTER_PROFILE_COLUMNS)
TRANSFORMER_FIGHT_FEATURE_SIZE = (
    len(STAT_COLUMNS) + len(FIGHT_METHOD_COLUMNS) + len(TRANSFORMER_OPPONENT_COLUMNS)
)

# Identity indices: unit-interval and categorical features (no z-score).
FEATURE_UNNORMALIZED_INDICES = identity_feature_indices(FEATURE_COLUMNS)
MATCHUP_UNNORMALIZED_INDICES = identity_feature_indices(MATCHUP_FEATURE_COLUMNS)
TRANSFORMER_UNNORMALIZED_INDICES = identity_feature_indices(TRANSFORMER_FEATURE_COLUMNS)
