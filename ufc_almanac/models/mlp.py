import torch
import torch.nn as nn

from ufc_almanac.globals import (
    INPUT_SIZE,
    NUM_CLASSES,
)


class MLPModel(nn.Module):

    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(features))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)
