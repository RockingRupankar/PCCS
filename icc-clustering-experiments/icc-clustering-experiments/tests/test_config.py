from dataclasses import asdict
from pathlib import Path
from icc import ICCConfig, load_icc_config

ROOT = Path(__file__).resolve().parents[1]


def test_yaml_defaults_match_code_defaults():
    assert asdict(load_icc_config(ROOT / "configs" / "icc_default.yaml")) == asdict(ICCConfig())


def test_overrides_applied():
    cfg = load_icc_config(ROOT / "configs" / "icc_default.yaml", n_views=8, n_clusters=3)
    assert cfg.n_views == 8 and cfg.n_clusters == 3


def test_pccs_yaml_defaults_match_code_defaults():
    from icc import PCCSConfig, load_pccs_config
    assert asdict(load_pccs_config(ROOT / "configs" / "pccs_default.yaml")) == asdict(PCCSConfig())
