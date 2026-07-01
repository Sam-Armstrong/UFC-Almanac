from .linear import LinearModel
from .mlp import MLPModel
from .transformer import TransformerModel

MODELS = {
    "linear": LinearModel,
    "mlp": MLPModel,
    "transformer": TransformerModel,
}
