"""Ingestion: zip -> clean PDFs -> normalised metadata -> chunks (LangChain Documents).

This is where the "messy data" requirements of the brief are handled:

* macOS junk (``__MACOSX/``, ``.DS_Store``, ``._*`` files) is skipped on extraction.
* Every policy uses a different header ("Policy ID" / "Document ID" / "Policy reference",
  "Last reviewed:" / "Last reviewed in", version sometimes missing...). We normalise all of
  them into the same metadata fields.
* Two different policies share the ID ``CS-POL-014``. IDs are therefore never used as a
  key: documents are identified by file name, and conflicting IDs are flagged so the
  assistant cites the document *title* instead of a misleading ID.
* The 2021 product catalog is much older than the rest of the corpus. Documents reviewed
  more than 12 months before the newest one are marked ``status="outdated"``; retrieval
  skips them unless the user explicitly asks about old data.
* SKUs reused by different products (e.g. ``IS-BED-001`` in two catalogs), identical
  files and duplicated chunks are detected and reported.

Run it with ``python -m ironstore_assistant.ingest`` to see the data-quality report
without building the index (no API key needed).
"""

from __future__ import annotations

import hashlib
import re
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from . import config
from .catalog import Product, parse_catalog

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}

# Header fields as they appear in the PDFs -> our normalised metadata key
HEADER_ALIASES = {
    "policy id": "doc_id",
    "document id": "doc_id",
    "policy reference": "doc_id",
    "version": "version",
    "last reviewed": "last_reviewed",
    "effective date": "effective_date",
    "owning area": "area_label",
    "policy owner": "owner",
    "document owner": "owner",
    "approved by": "approved_by",
    "applies to": "applies_to",
}
HEADER_RE = re.compile(r"^(?P<key>[A-Za-z ]+?)\s*(?::|\bin\b)\s*(?P<value>.+)$")


def is_junk(name: str) -> bool:
    """macOS metadata and other files that are not real documents."""
    parts = Path(name).parts
    return (
        "__MACOSX" in parts
        or any(p.startswith(".") for p in parts)
        or not name.lower().endswith(".pdf")
    )


def extract_zip(zip_path: Path = config.DOCUMENTS_ZIP, target: Path = config.EXTRACTED_DIR) -> list[Path]:
    """Extract only the real PDFs from the zip. Returns their paths."""
    target.mkdir(parents=True, exist_ok=True)
    root = target.resolve()
    out = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir() or is_junk(info.filename):
                continue
            dest = (target / info.filename).resolve()
            if root not in dest.parents:  # zip-slip protection
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(zf.read(info))
            out.append(dest)
    return sorted(out)


def parse_month_year(text: str) -> date | None:
    """'August 2026', '1 September 2026', 'December 2021' -> date."""
    m = re.search(r"(?:(\d{1,2})\s+)?([A-Za-z]+)\s+(\d{4})", text or "")
    if not m or m[2].lower() not in MONTHS:
        return None
    return date(int(m[3]), MONTHS[m[2].lower()], int(m[1] or 1))


@dataclass
class SourceDocument:
    path: Path
    pages: list[str]
    doc_type: str  # "policy" | "catalog"
    title: str
    area: str
    meta: dict = field(default_factory=dict)
    sha256: str = ""

    @property
    def source(self) -> str:
        return self.path.name


def _clean_page(text: str) -> str:
    # Bullets come out of pypdf as control characters (\x7f) or "■" (which also replaces
    # some non-latin letters, e.g. "Wroc■aw"); turn them into spaces.
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f■•]", " ", text)
    return "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()).strip()


def load_pdf(path: Path) -> SourceDocument:
    reader = PdfReader(path)
    pages = [_clean_page(p.extract_text() or "") for p in reader.pages]
    first_lines = [l.strip() for l in pages[0].splitlines() if l.strip()] if pages else []
    is_catalog = "product_catalog" in path.parts or path.name.startswith("IronStore_Product_Catalog") \
        or "Catalog" in path.name

    meta: dict = {}
    for line in first_lines[1:12]:
        m = HEADER_RE.match(line)
        if m and (key := HEADER_ALIASES.get(m["key"].strip().lower())):
            meta.setdefault(key, m["value"].strip())

    if is_catalog:
        category = first_lines[1] if len(first_lines) > 1 and not first_lines[1].startswith("Last") else ""
        if "Old" in path.name:
            category = "All categories (2021 edition)"
        title = f"IronStore Product Catalog — {category}".strip(" —")
        area = "Product catalog"
    else:
        title = first_lines[0] if first_lines else path.stem
        # The folder name is the only consistent department label
        # ("Document owner: HR_policies" is used as an area in one file).
        area = path.parent.name.replace("_", " ")

    doc = SourceDocument(path=path, pages=pages, doc_type="catalog" if is_catalog else "policy",
                         title=title, area=area, meta=meta)
    doc.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    reviewed = parse_month_year(meta.get("last_reviewed", ""))
    doc.meta["last_reviewed_date"] = reviewed.isoformat() if reviewed else ""
    return doc


@dataclass
class IngestResult:
    documents: list[Document]
    sources: list[SourceDocument]
    products: pd.DataFrame
    issues: dict[str, list[str]]


def _norm(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def ingest(pdf_paths: list[Path] | None = None) -> IngestResult:
    """Load every PDF and turn it into chunks ready to embed."""
    if pdf_paths is None:
        pdf_paths = extract_zip()
    issues: dict[str, list[str]] = defaultdict(list)

    # 1. Load, and drop byte-identical duplicates
    sources: list[SourceDocument] = []
    seen_hashes: dict[str, str] = {}
    for path in pdf_paths:
        doc = load_pdf(path)
        if doc.sha256 in seen_hashes:
            issues["Identical files (skipped)"].append(f"{doc.source} = {seen_hashes[doc.sha256]}")
            continue
        seen_hashes[doc.sha256] = doc.source
        sources.append(doc)

    # 2. Metadata problems
    by_id: dict[str, list[SourceDocument]] = defaultdict(list)
    for doc in sources:
        if doc.doc_type != "policy":
            continue
        if not doc.meta.get("doc_id"):
            issues["Policies without an ID"].append(doc.source)
        else:
            by_id[doc.meta["doc_id"]].append(doc)
        if not doc.meta.get("version"):
            issues["Policies without a version number"].append(doc.source)
    for doc_id, docs in by_id.items():
        if len(docs) > 1:
            titles = " / ".join(f"{d.title} ({d.source})" for d in docs)
            issues["Document ID shared by different documents"].append(f"{doc_id}: {titles}")
            for d in docs:
                d.meta["id_conflict"] = True

    # 3. Outdated documents: reviewed > 12 months before the newest document
    dated = [date.fromisoformat(d.meta["last_reviewed_date"]) for d in sources if d.meta.get("last_reviewed_date")]
    newest = max(dated) if dated else None
    for doc in sources:
        reviewed = doc.meta.get("last_reviewed_date")
        doc.meta["status"] = "current"
        if newest and reviewed and (newest - date.fromisoformat(reviewed)).days > 365:
            doc.meta["status"] = "outdated"
            issues["Outdated documents (excluded from answers by default)"].append(
                f"{doc.source}: last reviewed {doc.meta['last_reviewed']} (newest document: {newest:%B %Y})")

    # 4. Chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    chunks: list[Document] = []
    product_rows: list[dict] = []
    for doc in sources:
        base = {
            "source": doc.source,
            "title": doc.title,
            "doc_type": doc.doc_type,
            "area": doc.area,
            "doc_id": doc.meta.get("doc_id", ""),
            "version": doc.meta.get("version", ""),
            "last_reviewed": doc.meta.get("last_reviewed", ""),
            "effective_date": doc.meta.get("effective_date", ""),
            "status": doc.meta["status"],
            "id_conflict": bool(doc.meta.get("id_conflict")),
        }
        label = _label(base)
        if doc.doc_type == "policy":
            for page_no, text in enumerate(doc.pages, start=1):
                for piece in splitter.split_text(text):
                    chunks.append(Document(page_content=f"{label} | page {page_no}\n{piece}",
                                           metadata={**base, "page": page_no, "kind": "policy"}))
        else:
            notes, products = parse_catalog(doc.pages, doc.source)
            for page_no, text in notes:
                for piece in splitter.split_text(text):
                    chunks.append(Document(page_content=f"{label} | catalog notes | page {page_no}\n{piece}",
                                           metadata={**base, "page": page_no, "kind": "catalog_notes"}))
            for p in products:
                chunks.append(Document(
                    page_content=f"{label} | section: {p.section} | page {p.page}\n{p.text}",
                    metadata={**base, "page": p.page, "kind": "product", "sku": p.sku,
                              "product_name": p.name}))
                product_rows.append({**p.as_row(), "catalog": doc.title, "status": doc.meta["status"],
                                     "last_reviewed": base["last_reviewed"]})
            if not products:
                issues["Catalogs where no product could be parsed"].append(doc.source)

    # 5. Duplicated chunks (same normalised text, e.g. repeated boilerplate)
    unique, seen = [], {}
    for c in chunks:
        body = _norm(c.page_content.split("\n", 1)[-1])
        key = (c.metadata["source"], body)
        if key in seen:
            continue
        seen[key] = True
        unique.append(c)
    if dropped := len(chunks) - len(unique):
        issues["Duplicated chunks removed"].append(f"{dropped} chunks repeated verbatim inside the same document")
    for i, c in enumerate(unique):
        c.metadata["chunk_id"] = i

    # 6. Product-level problems
    products = pd.DataFrame(product_rows)
    if not products.empty:
        current = products[products.status == "current"]
        for sku, grp in current.groupby("sku"):
            if grp.source.nunique() > 1:
                where = "; ".join(f"{r.name} in {r.source} p.{r.page} (€{r.price_eur})" for r in grp.itertuples())
                issues["SKU reused by different products"].append(f"{sku}: {where}")
        missing = current[(current.availability == "") | current.price_eur.isna()]
        for r in missing.itertuples():
            issues["Products with missing price or availability"].append(f"{r.sku} {r.name} ({r.source} p.{r.page})")

    return IngestResult(documents=unique, sources=sources, products=products, issues=dict(issues))


def _label(meta: dict) -> str:
    """Short header prepended to every chunk so both the retriever and the LLM see it."""
    parts = [meta["title"]]
    if meta["doc_id"]:
        parts.append(f"ID {meta['doc_id']}" + (f" v{meta['version']}" if meta["version"] else ""))
    parts.append(meta["area"])
    if meta["last_reviewed"]:
        parts.append(f"last reviewed {meta['last_reviewed']}")
    if meta["status"] == "outdated":
        parts.append("OUTDATED")
    return "[" + " | ".join(parts) + "]"


def quality_report(result: IngestResult) -> str:
    """Markdown summary of the corpus and every data issue found."""
    policies = [s for s in result.sources if s.doc_type == "policy"]
    catalogs = [s for s in result.sources if s.doc_type == "catalog"]
    lines = [
        "# Data quality report",
        "",
        f"- Documents ingested: **{len(result.sources)}** ({len(policies)} policies, {len(catalogs)} catalogs)",
        f"- Pages: **{sum(len(s.pages) for s in result.sources)}**",
        f"- Chunks indexed: **{len(result.documents)}**",
        f"- Products parsed: **{len(result.products)}**",
        "",
        "## Policies",
        "",
        "| Area | Title | ID | Version | Last reviewed | Effective | File |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in sorted(policies, key=lambda s: (s.area, s.title)):
        m = s.meta
        doc_id = m.get("doc_id", "—") + (" ⚠️ shared" if m.get("id_conflict") else "")
        lines.append(f"| {s.area} | {s.title} | {doc_id} | {m.get('version', '—')} | "
                     f"{m.get('last_reviewed', '—')} | {m.get('effective_date', '—')} | {s.source} |")
    lines += ["", "## Issues found and how they are handled", ""]
    for name, items in result.issues.items():
        lines.append(f"### {name} ({len(items)})")
        lines.append("")
        lines += [f"- {i}" for i in items[:50]]
        if len(items) > 50:
            lines.append(f"- … and {len(items) - 50} more")
        lines.append("")
    return "\n".join(lines)


def save_outputs(result: IngestResult) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    result.products.to_csv(config.PRODUCTS_CSV, index=False)
    config.QUALITY_REPORT.write_text(quality_report(result), encoding="utf-8")


if __name__ == "__main__":
    res = ingest()
    save_outputs(res)
    print(quality_report(res))
