from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITES = ["exp0_falsification", "exp1_generalization", "exp2_ablation",
          "exp3_real_world_benchmarks", "exp4_scalability_and_sensitivity"]


def test_every_suite_has_required_parts():
    for s in SUITES:
        d = ROOT / "experiments" / s
        for part in ("data", "figures", "src", "README.md", "run_all.py"):
            assert (d / part).exists(), f"{s}/{part} missing"
        assert any((d / "figures").glob("*.png")), f"{s} has no figures"
        assert any((d / "data").glob("*.csv")), f"{s} has no data"


def test_all_python_files_compile():
    for f in list((ROOT / "experiments").rglob("*.py")) + list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").glob("*.py")):
        compile(f.read_text(encoding="utf-8"), str(f), "exec")


def test_root_files_present():
    for name in ("README.md", "LICENSE", "CITATION.cff", "CODE_OF_CONDUCT.md", "CONTRIBUTING.md", "SECURITY.md",
                 "pyproject.toml", "requirements.txt", "demo.py", ".gitignore", ".gitattributes"):
        assert (ROOT / name).exists(), name


def test_pccs_experiment_scripts_present():
    for rel in ("exp0_falsification/src/exp0_pccs_fingerprint.py", "exp2_ablation/src/exp2_pccs_ablation.py",
                "exp3_real_world_benchmarks/src/exp3_pccs.py", "exp4_scalability_and_sensitivity/src/exp4_pccs_response.py"):
        assert (ROOT / "experiments" / rel).exists(), rel
    assert (ROOT / "docs" / "pccs_traceability.md").exists()
