.PHONY: install test lint index app evaluate crawl merge

install:
	python -m pip install -c constraints/typecheck.txt -e ".[dev,crawl]"

test:
	pytest

lint:
	ruff check .
	ruff format --check .
	mypy

index:
	tcmb-rag build-index

app:
	streamlit run app/streamlit_app.py

evaluate:
	tcmb-rag evaluate

crawl:
	python scripts/tcmb_crawler.py --scope all --include-reference --output data/raw

merge:
	python scripts/merge_crawls.py \
		data/parts/a data/parts/b data/parts/c data/parts/d data/parts/v101_delta \
		--output data/raw
