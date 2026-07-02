import torch
from torch.utils.data import Dataset


class FightSequenceDataset(Dataset):

    def __init__(self, data: dict[str, torch.Tensor]):
        self.fighter1 = data["fighter1"]
        self.fighter2 = data["fighter2"]
        self.fighter1_mask = data["fighter1_mask"]
        self.fighter2_mask = data["fighter2_mask"]
        self.labels = data["labels"]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return (
            self.fighter1[index],
            self.fighter2[index],
            self.fighter1_mask[index],
            self.fighter2_mask[index],
            self.labels[index],
        )
