.PHONY: install lock pdf format lint type test security sample train evaluate drift serve docker-up docker-down clean

install:
	python -m pip install --require-hashes -r requirements-dev.lock
	python -m pip install --no-deps -e .

format:
	ruff format .
	ruff check --fix .

lint:
	ruff format --check .
	ruff check .

type:
	mypy src

test:
	pytest

security:
	bandit -c pyproject.toml -r src
	pip-audit -r requirements-dev.lock

lock:
	uv pip compile pyproject.toml --python-version 3.11 --python-platform \
	  x86_64-manylinux_2_28 --generate-hashes -o requirements.lock --no-header
	uv pip compile pyproject.toml --extra dev --python-version 3.11 --python-platform \
	  x86_64-manylinux_2_28 --generate-hashes -o requirements-dev.lock --no-header
	( echo 'reportlab>=4.2,<5' ) | uv pip compile - --python-version 3.11 --python-platform \
	  x86_64-manylinux_2_28 --generate-hashes -o requirements-docs.lock --no-header

pdf:
	python -m pip install --require-hashes -r requirements-docs.lock
	python tools/generate_pdf.py

sample:
	python scripts/generate_sample_data.py
	python scripts/generate_sample_data.py --start 2025-02-01 --seed 7 --output data/raw/sample_next.parquet

train:
	taxi-ml train data/raw/sample.parquet --model-version local-sample

evaluate:
	taxi-ml evaluate data/raw/sample_next.parquet

drift:
	taxi-ml drift data/raw/sample_next.parquet

serve:
	uvicorn taxi_duration.serving.app:app --reload --port 8000

docker-up:
	docker compose up --build

docker-down:
	docker compose down

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build
