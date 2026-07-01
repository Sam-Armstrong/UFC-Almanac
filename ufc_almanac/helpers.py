import torch


def get_device() -> torch.device:
    """
    Get the device to use for training and inference.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def resolve_model(
    model_name: str,
    models: dict[str, torch.nn.Module],
) -> torch.nn.Module:
    """
    Resolve the model class from the model name.
    """
    if model_name in models:
        return models[model_name]
    else:
        for model_class in models.values():
            if model_class.__name__ == model_name:
                return model_class

        raise ValueError(
            f"Unknown model {model_name!r}. "
            f"Available models: {', '.join(sorted(models))}"
        )
