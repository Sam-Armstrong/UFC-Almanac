class MinFightsException(Exception):
    """
    Exception raised when a fighter has had fewer than the minimum number of fights to be considered for training
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class MissingDataException(Exception):
    """
    Exception raised when one or more of the necessary data files is not present
    """

    def __init__(self):
        self.message = "One or more of the necessary data files is missing (FightResults.csv, FightStats.csv, and/or FighterData.csv)"
        super().__init__(self.message)
