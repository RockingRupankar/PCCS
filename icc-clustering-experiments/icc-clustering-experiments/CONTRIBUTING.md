# Contributing

Thank you for your interest in improving this project.

## Development setup

```bash
git clone https://github.com/AkashKundu114/<REPOSITORY-NAME>.git
cd <REPOSITORY-NAME>
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Workflow

1. Open an issue describing the bug or proposal.
2. Create a branch from `main`, make focused commits, and add or update tests in `tests/`.
3. Run `pytest` and `python demo.py` locally; the CI workflow runs the same checks on Python 3.10 to 3.12.
4. Open a pull request using the template and describe any change to results.

## Adding or changing an experiment

- Each suite lives in `experiments/<name>/` with `data/`, `figures/`, `src/`, `README.md`, and `run_all.py`.
- Use fixed seeds, write result tables to `data/` as CSV, and write figures to `figures/`.
- Record the settings in `configs/<name>.yaml` and update `docs/experiments.md`.
- Regenerate `docs/results_summary.md` with `python scripts/summarize_results.py`.
- If a change alters published numbers, update the README tables from `docs/results_summary.md` in the same pull request.

## Style

- Python 3.10+, type hints where they help, short docstrings that state purpose and units.
- Academic, neutral tone in documentation; report limitations next to results.
