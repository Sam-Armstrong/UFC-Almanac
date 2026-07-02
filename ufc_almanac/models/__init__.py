from .linear import LinearModel
from .mlp import MLPModel
from .transformer import TransformerModel

MODELS = {
    "linear": LinearModel,
    "mlp": MLPModel,
    "transformer": TransformerModel,
}

__all__ = ["LinearModel", "MLPModel", "TransformerModel", "MODELS"]
