.PHONY: install test run full notebooks clean-data

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

run:
	python scripts/run_pipeline.py

full:
	python scripts/run_pipeline.py --full

notebooks:
	python scripts/build_notebooks.py --execute

clean-data:
	python -c "from pathlib import Path; [p.unlink() for p in Path('data').glob('creditcard.*')]"

