import math
import torch
import torch.nn as nn

from ufc_almanac.globals import (
    TRANSFORMER_FEATURE_SIZE,
    MAX_FIGHTS,
    NUM_CLASSES,
)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = MAX_FIGHTS):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerModel(nn.Module):
    """
    Encodes each fighter's past fights with a shared transformer encoder,
    concatenates the pooled representations, and predicts win / loss / draw.
    """

    def __init__(
        self,
        TRANSFORMER_FEATURE_SIZE: int = TRANSFORMER_FEATURE_SIZE,
        max_fights: int = MAX_FIGHTS,
        num_classes: int = NUM_CLASSES,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_fights = max_fights
        self.input_proj = nn.Linear(TRANSFORMER_FEATURE_SIZE, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_fights)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model,
            dropout=dropout,
            batch_first=True,
            activation=nn.functional.gelu,
            bias=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes, bias=False),
        )

    def encode_fighter(
        self,
        fight_sequence: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.input_proj(fight_sequence)
        x = self.pos_encoder(x)
        mask_weights = mask.unsqueeze(-1).float()
        x = x * mask_weights
        x = self.transformer(x)
        return (x * mask_weights).sum(dim=1) / mask_weights.sum(dim=1).clamp(min=1.0)

    def forward(
        self,
        fighter1_fights: torch.Tensor,
        fighter2_fights: torch.Tensor,
        fighter1_mask: torch.Tensor,
        fighter2_mask: torch.Tensor,
    ) -> torch.Tensor:
        fighter1_embedding = self.encode_fighter(fighter1_fights, fighter1_mask)
        fighter2_embedding = self.encode_fighter(fighter2_fights, fighter2_mask)
        combined = torch.cat([fighter1_embedding, fighter2_embedding], dim=-1)
        return self.classifier(combined)
