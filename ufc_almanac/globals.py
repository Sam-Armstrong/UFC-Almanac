FIGHTER_DATA_CSV = "data/FighterData.csv"
RESULTS_CSV = "data/FightResults.csv"
STATS_CSV = "data/FightStats.csv"
STANDARD_TRAINING_DATA_PATH = "data/StandardTrainingData.pt"
TRANSFORMER_STANDARD_TRAINING_DATA_PATH = "data/TransformerTrainingData.pt"

MIN_FIGHTS = 3
MAX_FIGHTS = 8
VERBOSE = False

FEATURE_COLUMNS = [
    "Height 1",
    "Reach 1",
    "Age 1",
    "Knockdowns PM 1",
    "Gets Knocked Down PM 1",
    "Sig Strikes Landed PM 1",
    "Sig Strikes Attempted PM 1",
    "Sig Strikes Absorbed PM 1",
    "Strikes Landed PM 1",
    "Strikes Attempted PM 1",
    "Strikes Absorbed PM 1",
    "Strike Accuracy 1",
    "Takedowns PM 1",
    "Takedown Attempts PM 1",
    "Gets Taken Down PM 1",
    "Submission Attempts PM 1",
    "Clinch Strikes PM 1",
    "Clinch Strikes Taken PM 1",
    "Grounds Strikes PM 1",
    "Ground Strikes Taken PM 1",
    "Height 2",
    "Reach 2",
    "Age 2",
    "Knockdowns PM 2",
    "Gets Knocked Down PM 2",
    "Sig Strikes Landed PM 2",
    "Sig Strikes Attempted PM 2",
    "Sig Strikes Absorbed PM 2",
    "Strikes Landed PM 2",
    "Strikes Attempted PM 2",
    "Strikes Absorbed PM 2",
    "Strike Accuracy 2",
    "Takedowns PM 2",
    "Takedown Attempts PM 2",
    "Gets Taken Down PM 2",
    "Submission Attempts PM 2",
    "Clinch Strikes PM 2",
    "Clinch Strikes Taken PM 2",
    "Grounds Strikes PM 2",
    "Ground Strikes Taken PM 2",
]
LABEL_COLUMNS = ["Win", "Loss", "Draw"]
INPUT_SIZE = len(FEATURE_COLUMNS)
NUM_CLASSES = len(LABEL_COLUMNS)

TRANSFORMER_FEATURE_COLUMNS = [
    "Height",
    "Reach",
    "Age",
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
TRANSFORMER_FEATURE_SIZE = len(TRANSFORMER_FEATURE_COLUMNS)
