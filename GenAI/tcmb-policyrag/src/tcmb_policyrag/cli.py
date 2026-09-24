from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import typer

from .config import Settings
from .evaluation import evaluate_retrieval, load_cases
from .models import SearchFilters
from .service import RAGService

app = typer.Typer(no_args_is_help=True, help="TCMB PolicyRAG CLI")


@app.command("build-index")
def build_index(
    corpus: Path = typer.Option(Path("data/raw/documents.jsonl"), exists=True),
    max_tokens: int = 450,
    overlap_tokens: int = 70,
) -> None:
    settings = Settings.from_env()
    store = RAGService.build_index(corpus, settings, max_tokens, overlap_tokens)
    typer.echo(json.dumps(store.manifest, ensure_ascii=False, indent=2))


def filters(date_from: date | None, date_to: date | None, instrument: list[str]) -> SearchFilters:
    return SearchFilters(date_from=date_from, date_to=date_to, instruments=instrument)


def cli_date(value: str | None, option: str) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise typer.BadParameter(
            "YYYY-MM-DD biçiminde geçerli bir tarih girin", param_hint=option
        ) from exc


@app.command()
def search(
    query: str,
    top_k: int = 8,
    date_from: str | None = None,
    date_to: str | None = None,
    instrument: list[str] | None = typer.Option(None),
) -> None:
    search_filters = filters(
        cli_date(date_from, "--date-from"),
        cli_date(date_to, "--date-to"),
        instrument or [],
    )
    service = RAGService(Settings.from_env())
    hits = service.search(query, search_filters, top_k)
    typer.echo(json.dumps([hit.model_dump() for hit in hits], ensure_ascii=False, indent=2))


@app.command()
def ask(
    question: str,
    top_k: int = 8,
    date_from: str | None = None,
    date_to: str | None = None,
    instrument: list[str] | None = typer.Option(None),
) -> None:
    search_filters = filters(
        cli_date(date_from, "--date-from"),
        cli_date(date_to, "--date-to"),
        instrument or [],
    )
    service = RAGService(Settings.from_env())
    answer = service.ask(question, search_filters, top_k)
    typer.echo(answer.model_dump_json(indent=2))


@app.command()
def evaluate(
    benchmark: Path = typer.Option(Path("benchmarks/retrieval.json"), exists=True),
    k: int = 5,
    output: Path = Path("artifacts/evaluation/retrieval.json"),
) -> None:
    result = evaluate_retrieval(RAGService(Settings.from_env()), load_cases(benchmark), k)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
