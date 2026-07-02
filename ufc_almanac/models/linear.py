import torch
import torch.nn as nn

from ufc_almanac.globals import (
    INPUT_SIZE,
    NUM_CLASSES,
)


class LinearModel(nn.Module):

    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(self.dropout(features))
