import yaml
from pathlib import Path


class Config:
    """
    Lightweight dot-access configuration object.
    Turns nested dicts into attribute-accessible objects.
    """

    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)

    def to_dict(self):
        """Convert back to plain dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                value = value.to_dict()
            result[key] = value
        return result


def load_config(path: str | Path) -> Config:
    """
    Load YAML config file and return a Config object.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    return Config(data)


