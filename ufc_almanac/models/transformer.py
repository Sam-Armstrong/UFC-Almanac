import math
import torch
import torch.nn as nn

from ufc_almanac.globals import (
    TRANSFORMER_FEATURE_SIZE,
    MAX_FIGHTS,
    NUM_CLASSES,
)

DAYS_PER_YEAR = 365.25


def sinusoidal_encoding(values: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Apply sinusoidal encoding along the last dimension of values.

    Args:
        values: tensor whose last dimension is broadcast against d_model frequencies
        d_model: encoding width
    """
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=values.device, dtype=values.dtype)
        * (-math.log(10000.0) / d_model)
    )
    encoding = torch.zeros(*values.shape[:-1], d_model, device=values.device, dtype=values.dtype)
    encoding[..., 0::2] = torch.sin(values * div_term)
    encoding[..., 1::2] = torch.cos(values * div_term)
    return encoding


class TemporalPositionalEncoding(nn.Module):
    """
    Sinusoidal encoding driven by fight timing instead of sequence index.

    days_before is converted to years since the matchup; days_gap is converted to
    years between consecutive past fights. Each signal uses half of d_model.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.before_dims = d_model // 2
        self.gap_dims = d_model - self.before_dims

    def forward(
        self,
        x: torch.Tensor,
        days_before: torch.Tensor,
        days_gap: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        years_before = (days_before / DAYS_PER_YEAR).unsqueeze(-1)
        years_gap = (days_gap / DAYS_PER_YEAR).unsqueeze(-1)
        before_encoding = sinusoidal_encoding(years_before, self.before_dims)
        gap_encoding = sinusoidal_encoding(years_gap, self.gap_dims)
        temporal_encoding = torch.cat([before_encoding, gap_encoding], dim=-1)
        temporal_encoding = temporal_encoding * mask.unsqueeze(-1)
        return x + temporal_encoding


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
        self.pos_encoder = TemporalPositionalEncoding(d_model)
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
        days_before: torch.Tensor,
        days_gap: torch.Tensor,
    ) -> torch.Tensor:
        x = self.input_proj(fight_sequence)
        x = self.pos_encoder(x, days_before, days_gap, mask)
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
        fighter1_days_before: torch.Tensor,
        fighter2_days_before: torch.Tensor,
        fighter1_days_gap: torch.Tensor,
        fighter2_days_gap: torch.Tensor,
    ) -> torch.Tensor:
        fighter1_embedding = self.encode_fighter(
            fighter1_fights,
            fighter1_mask,
            fighter1_days_before,
            fighter1_days_gap,
        )
        fighter2_embedding = self.encode_fighter(
            fighter2_fights,
            fighter2_mask,
            fighter2_days_before,
            fighter2_days_gap,
        )
        combined = torch.cat([fighter1_embedding, fighter2_embedding], dim=-1)
        return self.classifier(combined)


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Scale logits by temperature for calibrated probability estimates.
    """
    if temperature == 1.0:
        return logits
    return logits / temperature
