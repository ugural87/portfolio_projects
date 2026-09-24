"""Crawl public TCMB Turkish policy announcements into auditable JSONL records.

Standard-library HTTP/HTML implementation; optional PyMuPDF for PDF text.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen

BASE = "https://www.tcmb.gov.tr"
ARCHIVE = BASE + "/wps/wcm/connect/TR/TCMB+TR/Main+Menu/Duyurular/Basin/{year}/"
SEEDS = {
    "reserve_ratios": BASE
    + "/wps/wcm/connect/TR/TCMB+TR/Main+Menu/Temel+Faaliyetler/Para+Politikasi/Zorunlu+Karsilik+Oranlari",
    "banking_legislation": BASE
    + "/wps/wcm/connect/TR/TCMB+TR/Main+Menu/Banka+Hakkinda/Mevzuat/Bankacilik",
    "banking_instructions": BASE
    + "/wps/wcm/connect/TR/TCMB+TR/Main+Menu/Banka+Hakkinda/Mevzuat/Bankacilik/Talimatlar",
}
INSTRUMENTS = {
    "reserve_requirements": r"zorunlu\s+karşılık|zorunlu\s+karsilik|rezerv\s+opsiyon|\brom\b",
    "securities_maintenance": r"menkul\s+kıymet\s+tesis|menkul\s+kiymet\s+tesis",
    "deposit_and_kkm": r"\bkkm\b|kur\s+korumal[ıi]|\bmevduat\w*|\byuvam\b|\bliralaş",
    "credit": r"\bkredi\w*|reeskont",
    "liquidity": r"likidite|sterilizasyon|swap|\brepo\w*",
    "macroprudential": r"makro\s*ihtiyati|makrofinansal|sadeleşme",
}
TITLE_POLICY = re.compile(
    "|".join(INSTRUMENTS.values()) + r"|sıkılaşma|sikilasma|dış yükümlülük|dis yukumluluk", re.I
)
DATE_TR = {
    "ocak": 1,
    "şubat": 2,
    "mart": 3,
    "nisan": 4,
    "mayıs": 5,
    "haziran": 6,
    "temmuz": 7,
    "ağustos": 8,
    "eylül": 9,
    "ekim": 10,
    "kasım": 11,
    "aralık": 12,
}
LOG = logging.getLogger("tcmb")


class Node:
    def __init__(self, tag="", attrs=(), parent=None):
        self.tag = tag
        self.attrs = dict(attrs)
        self.parent = parent
        self.children = []

    def walk(self):
        yield self
        for child in self.children:
            if isinstance(child, Node):
                yield from child.walk()

    def text(self):
        return "".join(c.text() if isinstance(c, Node) else c for c in self.children)


class Tree(HTMLParser):
    VOID = {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "wbr",
    }

    def __init__(self, raw):
        super().__init__(convert_charrefs=True)
        self.root = Node()
        self.cur = self.root
        self.feed(raw)

    def handle_starttag(self, tag, attrs):
        n = Node(tag, attrs, self.cur)
        self.cur.children.append(n)
        if tag not in self.VOID:
            self.cur = n

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in self.VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        n = self.cur
        while n is not self.root and n.tag != tag:
            n = n.parent
        if n is not self.root:
            self.cur = n.parent

    def handle_data(self, data):
        self.cur.children.append(data)


def compact(s):
    return re.sub(r"\s+", " ", s).strip()


def blocks(n):
    """Keep paragraphs/list boundaries and table rows for RAG source fidelity."""
    if n.tag in {"script", "style", "nav", "footer"}:
        return ""
    if n.tag == "table":
        rows = []
        for tr in n.walk():
            if tr.tag == "tr":
                cells = [
                    compact(c.text())
                    for c in tr.children
                    if isinstance(c, Node) and c.tag in {"td", "th"}
                ]
                if cells:
                    rows.append(" | ".join(cells))
        return "\n".join(rows) + "\n"
    if n.tag == "br":
        return "\n"
    inner = "".join(blocks(c) if isinstance(c, Node) else c for c in n.children)
    return inner + ("\n" if n.tag in {"p", "h1", "h2", "h3", "h4", "li", "div", "tr"} else "")


def clean_text(n):
    return "\n".join(compact(line) for line in blocks(n).splitlines() if compact(line))


def canonical(url):
    p = urlsplit(url)
    if p.scheme not in {"http", "https"} or not tcmb_host(p.hostname):
        raise ValueError(f"Non-TCMB URL: {url}")
    return urlunsplit(("https", "www.tcmb.gov.tr", p.path, "", ""))


def asset_url(url):
    p = urlsplit(url)
    if p.scheme != "https" or not tcmb_host(p.hostname):
        raise ValueError(f"Non-TCMB asset: {url}")
    return url


def tcmb_host(host):
    return bool(host and (host == "tcmb.gov.tr" or host.endswith(".tcmb.gov.tr")))


def parse_date(s):
    m = re.search(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b", s)
    if m:
        try:
            return date(int(m[3]), int(m[2]), int(m[1])).isoformat()
        except ValueError:
            pass
    m = re.search(r"\b(\d{1,2})\s+([a-zçğıöşü]+)\s+(\d{4})\b", s.casefold())
    if m and m[2] in DATE_TR:
        try:
            return date(int(m[3]), DATE_TR[m[2]], int(m[1])).isoformat()
        except ValueError:
            pass
    return None


def instrument_tags(s):
    return [k for k, pattern in INSTRUMENTS.items() if re.search(pattern, s, re.I)]


def links(tree, base):
    for n in tree.root.walk():
        if n.tag == "a" and n.attrs.get("href"):
            href = urljoin(base, html.unescape(n.attrs["href"]))
            if urlsplit(href).scheme in {"http", "https"}:
                yield n, href


def archive_entries(raw, url, year, excluded_documents=None, unparsed_boxes=None):
    excluded_documents = excluded_documents if excluded_documents is not None else []
    unparsed_boxes = unparsed_boxes if unparsed_boxes is not None else []
    tree = Tree(raw)
    results = {}
    for box in tree.root.walk():
        if "block-collection-box" not in box.attrs.get("class", "").split():
            continue
        candidates = list(links_from(box, url))
        title = next(
            (
                compact(node.text()) or node.attrs.get("title", "")
                for node, _href in candidates
                if compact(node.text()) or node.attrs.get("title")
            ),
            "",
        )
        dt = next(
            (
                parse_date(n.text())
                for n in box.walk()
                if n.tag == "div"
                and "collection-tag" in n.attrs.get("class", "").split()
                and parse_date(n.text())
            ),
            None,
        )
        pdfs = [
            asset_url(href)
            for _, href in candidates
            if urlsplit(href).path.lower().endswith(".pdf")
        ]
        pages = []
        for node, href in candidates:
            path = urlsplit(href).path
            # Most releases use DUY2024-12, while a small set of presidential
            # statements, interviews and presentations use descriptive slugs
            # such as DUY2020-Baskan. Both are first-class HTML announcements.
            if re.search(r"/Basin/\d{4}/DUY[^/]+/?$", path, re.I) and not path.lower().endswith(
                (".pdf", ".doc", ".docx", ".xls", ".xlsx", ".csv", ".zip")
            ):
                pages.append((node, href))
        if not pages:
            item = {
                "title": title,
                "date": dt,
                "archive_year": year,
                "archive_url": url,
                "links": sorted(set(href for _, href in candidates)),
            }
            if pdfs and "açık mektup" in title.casefold():
                item["reason"] = "open_letter_out_of_scope"
                excluded_documents.append(item)
                LOG.info("Excluded out-of-scope open letter: %s", title)
            elif pdfs and "sunum" in title.casefold():
                item["reason"] = "presentation_out_of_scope"
                excluded_documents.append(item)
                LOG.info("Excluded out-of-scope presentation: %s", title)
            else:
                item["reason"] = "no_supported_announcement_page"
                unparsed_boxes.append(item)
                LOG.warning("Unparsed archive box: %s", title or item["links"])
            continue
        node, page = pages[0]
        page = canonical(page)
        if dt and int(dt[:4]) != year:
            LOG.warning("Archive date/year conflict: %s %s", page, dt)
        results[page] = {
            "url": page,
            "title": compact(node.text()) or node.attrs.get("title", "") or title,
            "date": dt,
            "date_source": "archive" if dt else None,
            "archive_year": year,
            "archive_url": url,
            "archive_pdfs": sorted(set(pdfs)),
        }
    return list(results.values())


def links_from(node, base):
    for n in node.walk():
        if n.tag == "a" and n.attrs.get("href"):
            href = urljoin(base, html.unescape(n.attrs["href"]))
            if urlsplit(href).scheme in {"http", "https"}:
                yield n, href


def parse_page(raw, url, entry=None):
    entry = entry or {}
    tree = Tree(raw)
    nodes = list(tree.root.walk())
    h1 = next(
        (n for n in nodes if n.tag == "h1" and "block-header2-title" in n.attrs.get("class", "")),
        None,
    )
    main = next((n for n in nodes if n.attrs.get("id") == "tcmbMainContent"), None)
    if main is None:
        main = next(
            (n for n in nodes if "sub-home-content" in n.attrs.get("class", "").split()), None
        )
    if main is None:
        raise ValueError("TCMB content container not found; page layout may have changed")
    body = clean_text(main)
    if len(body) < 30:
        raise ValueError("Page content suspiciously short")
    title = compact(h1.text()) if h1 else entry.get("title", "")
    metas = [n.attrs for n in nodes if n.tag == "meta"]
    head_links = [n.attrs for n in nodes if n.tag == "link"]
    root_html = next((n for n in nodes if n.tag == "html"), None)
    published = parse_date(body[:350])
    date_value = published or entry.get("date")
    number = re.search(r"\b(?:Sayı\s*:\s*)?(20\d{2}-\d{1,3})\b", body[:300])
    if not number:
        number = re.search(r"DUY(20\d{2}-\d{1,3})", url, re.I)
    assets = []
    for a, href in links_from(main, url):
        if (
            urlsplit(href)
            .path.lower()
            .endswith((".pdf", ".doc", ".docx", ".xls", ".xlsx", ".csv", ".zip"))
        ):
            try:
                assets.append(
                    {
                        "url": asset_url(href),
                        "label": compact(a.text()) or a.attrs.get("title", ""),
                        "type": Path(urlsplit(href).path).suffix.lower(),
                    }
                )
            except ValueError:
                continue
    for pdf in entry.get("archive_pdfs", []):
        if pdf not in [a["url"] for a in assets]:
            assets.append({"url": pdf, "label": "Archive PDF", "type": ".pdf"})
    return {
        "id": "tcmb-" + hashlib.sha256(canonical(url).encode()).hexdigest()[:16],
        "source_url": canonical(url),
        "final_url": url,
        "source": "TCMB",
        "language": "tr",
        "document_type": "press_release" if "/Basin/" in url else "reference_page",
        "title": title,
        "announcement_number": number[1] if number else None,
        "published_date": date_value,
        "date_source": "body" if published else entry.get("date_source"),
        "archive_year": entry.get("archive_year"),
        "archive_url": entry.get("archive_url"),
        "policy_instruments": instrument_tags(title + "\n" + body),
        "is_policy_relevant": bool(TITLE_POLICY.search(title + "\n" + body)),
        "text": body,
        "links": sorted(
            set(href for _, href in links_from(main, url) if tcmb_host(urlsplit(href).hostname))
        ),
        "attachments": assets,
        "html_metadata": metas,
        "html_head_links": head_links,
        "html_language": root_html.attrs.get("lang") if root_html else None,
        "archive_date": entry.get("date"),
        "date_mismatch": bool(published and entry.get("date") and published != entry["date"]),
        "content_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "fetched_at_utc": datetime.now(UTC).isoformat(),
    }


class Client:
    def __init__(self, delay=1.2, timeout=25, retries=3, max_bytes=25_000_000):
        self.delay, self.timeout, self.retries, self.max_bytes = delay, timeout, retries, max_bytes
        self.last = 0.0
        self.lock = threading.Lock()

    def get(self, url):
        asset_url(url)  # SSRF guard, including caller-provided seeds
        for attempt in range(self.retries + 1):
            with self.lock:
                gap = self.delay - (time.monotonic() - self.last)
                if gap > 0:
                    time.sleep(gap)
                self.last = time.monotonic()
            try:
                with urlopen(
                    Request(
                        url,
                        headers={
                            "User-Agent": "TCMB-PolicyRAG-Research/1.0 (public academic archive)",
                            "Accept": "text/html,application/pdf,*/*",
                        },
                    ),
                    timeout=self.timeout,
                ) as r:
                    asset_url(r.url)  # reject off-domain redirects
                    length = r.headers.get("Content-Length")
                    if length and int(length) > self.max_bytes:
                        raise ValueError("Response exceeds max bytes")
                    data = r.read(self.max_bytes + 1)
                    if len(data) > self.max_bytes:
                        raise ValueError("Response exceeds max bytes")
                    return data, r.url, dict(r.headers)
            except (HTTPError, URLError, TimeoutError) as exc:
                if isinstance(exc, HTTPError) and exc.code not in {429, 500, 502, 503, 504}:
                    raise
                if attempt >= self.retries:
                    raise
                wait = min(30, 2**attempt + random.uniform(0, 0.5))
                LOG.warning("Retry %s for %s: %s", attempt + 1, url, exc)
                time.sleep(wait)


def decode(data, headers):
    charset = re.search(r"charset=([\w-]+)", headers.get("Content-Type", ""), re.I)
    return data.decode(charset[1] if charset else "utf-8", errors="replace")


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def file_key(url):
    return hashlib.sha256(url.encode()).hexdigest()[:20]


def portable_paths(record, out):
    prefix = str(out) + "/"
    if isinstance(record.get("raw_html_path"), str) and record["raw_html_path"].startswith(prefix):
        record["raw_html_path"] = record["raw_html_path"][len(prefix) :]
    for asset in record.get("attachments", []):
        for key in ("local_path", "pdf_text_path"):
            if isinstance(asset.get(key), str) and asset[key].startswith(prefix):
                asset[key] = asset[key][len(prefix) :]
    return record


def download_asset(client, item, folder, extract_pdf=False):
    url = item["url"]
    suffix = item["type"]
    path = folder / (file_key(url) + suffix)
    if path.exists():
        data = path.read_bytes()
    else:
        data, final, headers = client.get(url)
        if suffix == ".pdf" and not data.startswith(b"%PDF"):
            raise ValueError("PDF link returned non-PDF content")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        item["response_metadata"] = {
            k.lower(): v
            for k, v in headers.items()
            if k.lower() in {"content-type", "last-modified", "etag", "content-length"}
        }
        item["final_url"] = final
    item.update(
        {
            "local_path": str(path.relative_to(folder.parent)),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    )
    if suffix == ".pdf" and extract_pdf:
        try:
            import fitz
        except ImportError:
            LOG.warning("Install PyMuPDF to extract PDF text: %s", url)
        else:
            with fitz.open(stream=data, filetype="pdf") as pdf:
                pages = [
                    {"page": i + 1, "text": page.get_text(sort=True)} for i, page in enumerate(pdf)
                ]
            text_path = folder.parent / "pdf_text" / (file_key(url) + ".json")
            write_json(text_path, {"source_url": url, "pages": pages})
            item["pdf_text_path"] = str(text_path.relative_to(folder.parent))
            item["page_count"] = len(pages)
            item["text_page_count"] = sum(bool(p["text"].strip()) for p in pages)


def crawl(args):
    out = Path(args.output).resolve()
    client = Client(args.delay, args.timeout, args.retries, args.max_bytes)
    today = date.today()
    end = args.end_year or today.year
    if args.start_year < 2019 or end < args.start_year or end > today.year:
        raise ValueError("Use years between 2019 and the current year")
    entries = {}
    errors = []
    excluded_documents = []
    unparsed_boxes = []
    for year in range(args.start_year, end + 1):
        url = ARCHIVE.format(year=year)
        try:
            data, final, headers = client.get(url)
            items = archive_entries(
                decode(data, headers),
                final,
                year,
                excluded_documents=excluded_documents,
                unparsed_boxes=unparsed_boxes,
            )
            if not items:
                raise ValueError("No archive entries found; inspect layout")
            entries.update({e["url"]: e for e in items})
            LOG.info("%s: discovered %s releases", year, len(items))
        except Exception as exc:
            errors.append({"url": url, "stage": "archive", "error": str(exc)})
            LOG.error("Archive failed %s: %s", url, exc)
    if not entries:
        raise RuntimeError("No archive pages could be fetched")
    write_json(
        out / "archive_index.json",
        {
            "start_year": args.start_year,
            "end_year": end,
            "discovered_count": len(entries),
            "excluded_count": len(excluded_documents),
            "excluded_documents": excluded_documents,
            "unparsed_archive_boxes": unparsed_boxes,
            "entries": sorted(entries.values(), key=lambda e: (e["date"] or "", e["url"])),
        },
    )
    selected = [
        e for e in entries.values() if args.scope == "all" or TITLE_POLICY.search(e["title"])
    ]
    if args.url_regex:
        url_filter = re.compile(args.url_regex, re.I)
        selected = [e for e in selected if url_filter.search(e["url"])]
    selected.sort(key=lambda e: (e["date"] or "", e["url"]))
    if args.limit:
        selected = selected[: args.limit]

    def fetch_entry(entry):
        url = entry["url"]
        path = out / "records" / (file_key(url) + ".json")
        local_errors = []
        try:
            if path.exists() and not args.refresh:
                record = json.loads(path.read_text(encoding="utf-8"))
            else:
                raw, final, headers = client.get(url)
                decoded = decode(raw, headers)
                record = parse_page(decoded, final, entry)
                record["content_sha256"] = hashlib.sha256(raw).hexdigest()
                record["response_metadata"] = {
                    k.lower(): v
                    for k, v in headers.items()
                    if k.lower() in {"content-type", "last-modified", "etag", "content-length"}
                }
                html_path = out / "raw_html" / (file_key(url) + ".html")
                html_path.parent.mkdir(parents=True, exist_ok=True)
                html_path.write_bytes(raw)
                record["raw_html_path"] = str(html_path.relative_to(out))
            if args.download_assets:
                for asset in record["attachments"]:
                    try:
                        download_asset(client, asset, out / "assets", args.extract_pdf)
                    except Exception as exc:
                        asset["download_error"] = str(exc)
                        local_errors.append(
                            {"url": asset["url"], "stage": "asset", "error": str(exc)}
                        )
            portable_paths(record, out)
            write_json(path, record)
            return record, local_errors
        except Exception as exc:
            local_errors.append({"url": url, "stage": "page", "error": str(exc)})
            LOG.error("Page failed %s: %s", url, exc)
            return None, local_errors

    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (record, page_errors) in enumerate(pool.map(fetch_entry, selected), 1):
            errors.extend(page_errors)
            if record:
                records.append(record)
                LOG.info("%s/%s %s", i, len(selected), record["title"])
    if args.include_reference:
        for name, url in SEEDS.items():
            try:
                raw, final, headers = client.get(url)
                record = parse_page(decode(raw, headers), final)
                html_path = out / "raw_html" / (file_key(url) + ".html")
                html_path.parent.mkdir(parents=True, exist_ok=True)
                html_path.write_bytes(raw)
                record["raw_html_path"] = str(html_path.relative_to(out))
                record["content_sha256"] = hashlib.sha256(raw).hexdigest()
                record["response_metadata"] = {
                    k.lower(): v
                    for k, v in headers.items()
                    if k.lower() in {"content-type", "last-modified", "etag", "content-length"}
                }
                record["reference_name"] = name
                record["published_date"] = None  # live reference page is not a historical version
                record["date_source"] = None
                path = out / "records" / (file_key(url) + ".json")
                write_json(path, record)
                records.append(record)
            except Exception as exc:
                errors.append({"url": url, "stage": "reference", "error": str(exc)})
    export = out / "documents.jsonl"
    with export.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_json(
        out / "crawl_report.json",
        {
            "discovered": len(entries),
            "selected": len(selected),
            "exported": len(records),
            "failures": len(errors),
            "errors": errors,
            "excluded_documents": excluded_documents,
            "unparsed_archive_boxes": unparsed_boxes,
            "scope": args.scope,
            "discovered_by_year": dict(
                sorted(Counter(e["archive_year"] for e in entries.values()).items())
            ),
            "exported_by_year": dict(
                sorted(
                    Counter(
                        r["published_date"][:4] if r["published_date"] else "unknown"
                        for r in records
                    ).items()
                )
            ),
            "instrument_counts": dict(
                sorted(Counter(tag for r in records for tag in r["policy_instruments"]).items())
            ),
            "date_mismatch_count": sum(r["date_mismatch"] for r in records),
            "unknown_date_count": sum(not r["published_date"] for r in records),
            "run_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    LOG.info("Exported %s documents, %s errors: %s", len(records), len(errors), export)
    return 0 if not errors and not unparsed_boxes else 2


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="data/tcmb")
    p.add_argument("--start-year", type=int, default=2019)
    p.add_argument("--end-year", type=int, default=None)
    p.add_argument(
        "--scope",
        choices=["policy", "all"],
        default="policy",
        help="policy: title-filtered announcements; all: every press announcement",
    )
    p.add_argument("--download-assets", action="store_true")
    p.add_argument(
        "--extract-pdf", action="store_true", help="requires --download-assets and PyMuPDF"
    )
    p.add_argument(
        "--include-reference",
        action="store_true",
        help="snapshot current ratios, legislation and instructions",
    )
    p.add_argument("--refresh", action="store_true")
    p.add_argument(
        "--url-regex",
        help="fetch only discovered announcement URLs matching this regular expression",
    )
    p.add_argument(
        "--limit", type=int, default=None, help="smoke test: first N selected announcements"
    )
    p.add_argument("--delay", type=float, default=1.2)
    p.add_argument(
        "--workers",
        type=int,
        default=3,
        help="concurrent downloads; delay is shared between requests",
    )
    p.add_argument("--timeout", type=float, default=25)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--max-bytes", type=int, default=25_000_000)
    args = p.parse_args(argv)
    if args.extract_pdf and not args.download_assets:
        p.error("--extract-pdf requires --download-assets")
    if args.limit is not None and args.limit <= 0:
        p.error("--limit must be positive")
    if not 1 <= args.workers <= 8:
        p.error("--workers must be between 1 and 8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return crawl(args)


if __name__ == "__main__":
    raise SystemExit(main())
