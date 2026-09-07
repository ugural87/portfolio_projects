from __future__ import annotations

import re
import unicodedata


PUNCTUATION_TRANSLATION = str.maketrans(
    {"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-"}
)
LINE_WRAP_HYPHEN_PATTERN = re.compile(r"(?<=[A-Za-z])-\s*\n\s*(?=[a-z])")
FOMC_PAGE_BREAK_HEADER_PATTERN = re.compile(
    r"(?P<left>[A-Za-z])-\s+"
    r"(?i:Minutes)\s+of\s+the\s+(?i:Meeting)\s+of"
    r"[\s\S]{0,120}?"
    r"(?i:Page)\s+\d+\s+"
    r"(?P<right>[a-z])"
)
FED_SPACED_ELLIPSIS_PATTERN = re.compile(r'["\']?\s*(?:\.\s+){2}\.')
SENTENCE_CATALOG_VERSION = "canonical-sentence-catalog-v1"
SENTENCE_ID_WIDTH = 5
SENTENCE_ID_PATTERN = re.compile(r"^[PC]\d{5}$")


def canonicalise_spacing(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).translate(PUNCTUATION_TRANSLATION)
    return " ".join(value.split())


def normalise_evidence(value: str) -> str:
    return canonicalise_spacing(value).strip().strip("\"'").casefold()


def remove_fomc_page_break_headers(value: str) -> str:
    return FOMC_PAGE_BREAK_HEADER_PATTERN.sub(
        lambda match: f"{match.group('left')}{match.group('right')}", value
    )


def source_line_wrap_variants(value: str) -> tuple[str, str, str]:
    canonical = unicodedata.normalize("NFKC", value).translate(PUNCTUATION_TRANSLATION)
    cleaned = remove_fomc_page_break_headers(canonical)
    return canonical, LINE_WRAP_HYPHEN_PATTERN.sub("", cleaned), LINE_WRAP_HYPHEN_PATTERN.sub("-", cleaned)


def normalise_fed_editorial_ellipsis(value: str) -> str:
    return normalise_evidence(FED_SPACED_ELLIPSIS_PATTERN.sub(" ", canonicalise_spacing(value)))


def split_document_sentences_canonical(document: str) -> list[str]:
    protected = canonicalise_spacing(source_line_wrap_variants(document)[1])
    protected = re.sub(
        r"\b(?:[A-Za-z]\.){2,}", lambda match: match.group(0).replace(".", "<DOT>"), protected
    )
    for abbreviation in ("Mr.", "Mrs.", "Ms.", "Dr.", "e.g.", "i.e."):
        protected = re.sub(
            re.escape(abbreviation), abbreviation.replace(".", "<DOT>"), protected, flags=re.I
        )
    protected = re.sub(r"\bNo\.(?=\s+\d)", "No<DOT>", protected, flags=re.I)
    candidates = re.split(r"(?<=[.!?])\s+(?=[\"']?[A-Z0-9])", protected)
    return [candidate.replace("<DOT>", ".").strip() for candidate in candidates if candidate.strip()]


def build_sentence_catalog(document: str, prefix: str) -> dict[str, str]:
    if prefix not in {"P", "C"}:
        raise ValueError("Sentence-catalog prefix must be P or C.")
    sentences = split_document_sentences_canonical(document)
    if not sentences or len(sentences) >= 10**SENTENCE_ID_WIDTH:
        raise ValueError("Invalid sentence-catalog size.")
    return {
        f"{prefix}{index:0{SENTENCE_ID_WIDTH}d}": sentence
        for index, sentence in enumerate(sentences, start=1)
    }


def format_sentence_catalog(catalog: dict[str, str]) -> str:
    return "\n".join(f"[{key}] {sentence}" for key, sentence in catalog.items())


def resolve_evidence_id(evidence_id: str, catalog: dict[str, str], prefix: str) -> str:
    if not evidence_id:
        return ""
    if not SENTENCE_ID_PATTERN.fullmatch(evidence_id) or not evidence_id.startswith(prefix):
        raise ValueError(f"Invalid {prefix}-catalog evidence ID: {evidence_id}")
    if evidence_id not in catalog:
        raise KeyError(f"Evidence ID is absent from the supplied catalog: {evidence_id}")
    return catalog[evidence_id]
