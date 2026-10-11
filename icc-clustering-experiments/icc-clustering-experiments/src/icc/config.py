"""Load an ICCConfig from a YAML file (see configs/icc_default.yaml)."""
from pathlib import Path
import yaml
from .model import ICCConfig


def load_icc_config(path, **overrides):
    """Read the `icc:` section of a YAML file, convert list values to tuples, apply overrides."""
    with open(Path(path), "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    params = dict(raw.get("icc", raw))
    for key in ("families", "base_methods"):
        if key in params:
            params[key] = tuple(params[key])
    params.update(overrides)
    return ICCConfig(**params)


def load_pccs_config(path, **overrides):
    """Read the `pccs:` section of a YAML file (see configs/pccs_default.yaml) into a PCCSConfig."""
    from .pccs import PCCSConfig
    with open(Path(path), "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    params = dict(raw.get("pccs", raw))
    for key in ("methods", "families"):
        if key in params:
            params[key] = tuple(params[key])
    if "levels" in params:
        params["levels"] = {k: tuple(v) for k, v in params["levels"].items()}
    if params.get("interaction_pairs"):
        params["interaction_pairs"] = tuple(tuple(p) for p in params["interaction_pairs"])
    params.update(overrides)
    return PCCSConfig(**params)
