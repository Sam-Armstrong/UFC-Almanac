import math
import torch
import torch.nn as nn

from ufc_almanac.globals import (
    MATCHUP_FEATURE_SIZE,
    MAX_FIGHTS,
    NUM_CLASSES,
    TRANSFORMER_FIGHT_FEATURE_SIZE,
    TRANSFORMER_STATIC_FEATURE_SIZE,
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


class AttentionPooling(nn.Module):
    """
    Learned attention pooling over a masked fight sequence.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Linear(d_model, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(x).squeeze(-1)
        scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = weights * mask
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return (x * weights.unsqueeze(-1)).sum(dim=1)


class FighterCrossAttention(nn.Module):
    """
    Cross-attend one fighter sequence against the other fighter's sequence.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
            bias=False,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query_seq: torch.Tensor,
        query_mask: torch.Tensor,
        context_seq: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        key_padding_mask = context_mask == 0
        attended, _ = self.attention(
            query_seq,
            context_seq,
            context_seq,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        attended = self.dropout(attended)
        output = self.norm(query_seq + attended)
        return output * query_mask.unsqueeze(-1)


class TransformerModel(nn.Module):
    """
    Encode each fighter's past fights with a shared transformer encoder,
    apply cross-attention between fighters, pool with learned attention,
    and classify using explicit interaction features.
    """

    def __init__(
        self,
        static_feature_size: int = TRANSFORMER_STATIC_FEATURE_SIZE,
        fight_feature_size: int = TRANSFORMER_FIGHT_FEATURE_SIZE,
        max_fights: int = MAX_FIGHTS,
        num_classes: int = NUM_CLASSES,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_fights = max_fights
        self.static_feature_size = static_feature_size
        self.fight_feature_size = fight_feature_size
        self.static_proj = nn.Linear(static_feature_size, d_model)
        self.fight_proj = nn.Linear(fight_feature_size, d_model)
        self.pos_encoder = TemporalPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
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
        self.cross_attention = FighterCrossAttention(d_model, nhead, dropout)
        self.pooling = AttentionPooling(d_model)
        classifier_input_size = d_model * 4 + MATCHUP_FEATURE_SIZE
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, d_model, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes, bias=False),
        )

    def encode_sequence(
        self,
        fight_sequence: torch.Tensor,
        mask: torch.Tensor,
        days_before: torch.Tensor,
        days_gap: torch.Tensor,
    ) -> torch.Tensor:
        static_features = fight_sequence[:, 0, : self.static_feature_size]
        fight_features = fight_sequence[:, :, self.static_feature_size :]
        x = self.fight_proj(fight_features) + self.static_proj(static_features).unsqueeze(1)
        x = self.pos_encoder(x, days_before, days_gap, mask)
        x = x * mask.unsqueeze(-1)
        padding_mask = mask == 0
        return self.transformer(x, src_key_padding_mask=padding_mask)

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
        matchup_features: torch.Tensor,
    ) -> torch.Tensor:
        encoded1 = self.encode_sequence(
            fighter1_fights,
            fighter1_mask,
            fighter1_days_before,
            fighter1_days_gap,
        )
        encoded2 = self.encode_sequence(
            fighter2_fights,
            fighter2_mask,
            fighter2_days_before,
            fighter2_days_gap,
        )
        crossed1 = self.cross_attention(
            encoded1,
            fighter1_mask,
            encoded2,
            fighter2_mask,
        )
        crossed2 = self.cross_attention(
            encoded2,
            fighter2_mask,
            encoded1,
            fighter1_mask,
        )
        fighter1_embedding = self.pooling(crossed1, fighter1_mask)
        fighter2_embedding = self.pooling(crossed2, fighter2_mask)
        combined = torch.cat(
            [
                fighter1_embedding,
                fighter2_embedding,
                fighter1_embedding - fighter2_embedding,
                fighter1_embedding * fighter2_embedding,
                matchup_features,
            ],
            dim=-1,
        )
        return self.classifier(combined)


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Scale logits by temperature for calibrated probability estimates.
    """
    if temperature == 1.0:
        return logits
    return logits / temperature
