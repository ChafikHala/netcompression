import torch


def get_device(device_str: str | None = "auto") -> torch.device:
    """
    Resolve device automatically.

    device_str:
        "auto"  -> use cuda if available else cpu
        "cuda"  -> force cuda (error if unavailable)
        "cpu"   -> cpu
        None    -> same as "auto"
    """

    if device_str is None or device_str.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device_str = device_str.lower()

    if device_str == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")

    if device_str == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unknown device option: {device_str}")
