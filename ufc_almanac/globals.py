import os


FIGHTER_DATA_CSV = "data/FighterData.csv"
RESULTS_CSV = "data/FightResults.csv"
STATS_CSV = "data/FightStats.csv"
STANDARD_TRAINING_DATA_PATH = "data/StandardTrainingData.pt"
TRANSFORMER_STANDARD_TRAINING_DATA_PATH = "data/TransformerTrainingData.pt"
CHECKPOINTS_DIR = "artifacts/checkpoints"

BLOCKED_RESOURCE_TYPES = {"image", "media", "font"}
MIN_FIGHTS = 3
MAX_FIGHTS = 8
VERBOSE = bool(int(os.getenv("VERBOSE", 0)))
WEIGHT_CLASS_MISMATCH_THRESHOLD = 10
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
]

FIGHTER_PROFILE_COLUMNS = [
    "Height",
    "Reach",
    "Age",
    "Weight",
    "Stance",
]

FIGHTER_FORM_COLUMNS = [
    "Recent Win Rate",
    "Avg Opponent Win Rate",
    "Days Since Last Fight",
]

FIGHTER_FEATURE_COLUMNS = (
    FIGHTER_PROFILE_COLUMNS
    + STAT_COLUMNS
    + FIGHTER_FORM_COLUMNS
)

MATCHUP_FEATURE_COLUMNS = [
    "Reach Diff",
    "Height Diff",
    "Age Diff",
    "Weight Diff",
    "Weight Class Mismatch",
    "Stance Mismatch",
    "Days Since Last Fight 1",
    "Days Since Last Fight 2",
]

FEATURE_COLUMNS = (
    [f"{column} 1" for column in FIGHTER_FEATURE_COLUMNS]
    + [f"{column} 2" for column in FIGHTER_FEATURE_COLUMNS]
    + MATCHUP_FEATURE_COLUMNS
)
LABEL_COLUMNS = ["Win", "Loss", "Draw"]
INPUT_SIZE = len(FEATURE_COLUMNS)
NUM_CLASSES = len(LABEL_COLUMNS)
MATCHUP_FEATURE_SIZE = len(MATCHUP_FEATURE_COLUMNS)

TRANSFORMER_OPPONENT_COLUMNS = [
    "Opponent Height",
    "Opponent Reach",
    "Opponent Age",
    "Opponent Weight",
    "Opponent Win Rate",
    "Fight Outcome",
]

TRANSFORMER_FEATURE_COLUMNS = (
    FIGHTER_PROFILE_COLUMNS
    + STAT_COLUMNS
    + TRANSFORMER_OPPONENT_COLUMNS
)
TRANSFORMER_FEATURE_SIZE = len(TRANSFORMER_FEATURE_COLUMNS)
