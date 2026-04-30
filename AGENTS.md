# Repository Guidelines

## Project Structure & Module Organization
Core package code lives in `py/dynesty/` (samplers, bounds, utilities, plotting, and results APIs).  
Tests are in `tests/` and use `pytest` with strict markers.  
Documentation sources are in `docs/source/`, built via the Sphinx `docs/Makefile`.  
Notebook demos are in `demos/`; the paper artifact is in `paper/`.  
Packaging and tool config are in `pyproject.toml`, `pytest.ini`, and `.pylintrc`.

## Build, Test, and Development Commands
- `pip install .[dev]`: install dynesty plus development/test/docs dependencies.
- `pytest --durations=0 -m "not slow"`: run the standard fast CI-style suite.
- `pytest --durations=0 -m "not slow" --cov=dynesty`: run coverage (used in CI Python 3.10 job).
- `pytest tests/test_notebooks.py`: validate Jupyter notebooks (slow; requires working Jupyter kernel).
- `pylint --fail-under=9 --ignored-modules=scipy.special py/dynesty/*py`: run linting at CI threshold.
- `cd docs && make html SPHINXOPTS="-W --keep-going"`: build docs and fail on warnings.

## Coding Style & Naming Conventions
Use Python style with 4-space indentation and clear, focused functions.  
Follow existing module patterns in `py/dynesty/`; keep public API names stable unless explicitly planned.  
`pylint` is enforced (`--fail-under=9`), with repository-specific exceptions already configured.  
For randomness in tests and internals, avoid ad-hoc `np.random.*` usage; follow existing `Generator/default_rng` conventions.

## Testing Guidelines
Add or update tests for every behavioral change in `tests/test_*.py`.  
Do not reduce overall coverage (project guidance targets roughly 91%+).  
Mark long-running tests with `@pytest.mark.slow`; keep default test runs practical.  
Use `tests/utils.py:get_rstate()` and `DYNESTY_RANDOM_SEED` for deterministic test behavior.

## Commit & Pull Request Guidelines
Recent commit history favors short, imperative subjects (for example: `fix versions`, `clean up docs`, `test python 3.14`).  
Keep commit messages concise and specific to one logical change.  
PRs should include: purpose, linked issue (if any), reproduction details for bug fixes, and tests demonstrating the change.  
If behavior or interfaces change, update docs and changelog entries in the same PR.
