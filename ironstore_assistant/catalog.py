"""Split product catalogs into one record per product.

The 63 catalogs were clearly written by different people: we found at least six layouts
("1. Name" + "SKU:" line, "CODE — Name", "Product title:", "Product:", indented fields,
discontinued sections...). What they all share is a ``Category:`` line inside every
product, so we use it as the anchor and walk backwards to find where the product starts.

Each product becomes:
- one chunk for the vector store (so a retrieved chunk never mixes two products), and
- one row of a structured table (SKU, price, availability...) used by the exact
  ``lookup_product`` tool, because embeddings are bad at matching exact SKUs and prices.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

# "Page 3 — Sofas" or "Page 2 of 15 — Bed Linen Sets": the catalog's own section header
SECTION_RE = re.compile(r"^Page\s+\d+(?:\s+of\s+\d+)?\s*[—-]\s*(?P<section>.+)$")
CATEGORY_RE = re.compile(r"^Category\s*:")
SKU_LINE_RE = re.compile(r"^SKU\s*:\s*(?P<sku>\S+)")
TITLE_FIELD_RE = re.compile(r"^Product(?: title)?\s*:\s*(?P<name>.+)$")
# "IS-BED-001 — Nordhaven Percale Bedding Set" (code first, then the name)
CODE_TITLE_RE = re.compile(r"^(?P<sku>[A-Z]{2,}[A-Z0-9]*(?:-[A-Z0-9]+)+)\s+[—-]\s+(?P<name>.+)$")
# Fields that some layouts put between the product title and its Category line
PRE_CATEGORY_RE = re.compile(r"^(Brand|Product status)\s*:")
NUMBERED_RE = re.compile(r"^\d{1,3}\.\s+")
DISCONTINUED_MARK = "This product has been discontinued"
PRICE_RE = re.compile(r"€\s?(?P<amount>\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)")
FIELD_RE = re.compile(r"^(?P<key>[A-Z][A-Za-z /]{1,30}?)\s*:\s*(?P<value>.*)$")


@dataclass
class Product:
    sku: str
    name: str
    category: str = ""
    price_eur: float | None = None
    price_text: str = ""
    availability: str = ""
    warehouse: str = ""
    replacement_sku: str = ""
    section: str = ""
    page: int = 0  # 1-based PDF page where the product starts
    source: str = ""  # file name, filled by the caller
    text: str = field(default="", repr=False)

    def as_row(self) -> dict:
        row = asdict(self)
        row.pop("text")
        return row


def _lines_with_pages(pages: list[str]) -> list[tuple[int, str]]:
    out = []
    for page_no, text in enumerate(pages, start=1):
        for raw in text.splitlines():
            line = raw.strip()
            if line:
                out.append((page_no, line))
    return out


def _find_start(lines: list[tuple[int, str]], cat_idx: int, floor: int) -> tuple[int, str, str]:
    """Return (start index, sku, name) for the product whose Category line is cat_idx."""
    sku, name, start = "", "", cat_idx
    # Short fields (SKU, Brand, Product status, Product title...) and the "discontinued"
    # banner may sit between the title and the Category line.
    j = cat_idx - 1
    while j >= floor and j >= cat_idx - 4:
        line = lines[j][1]
        if m := SKU_LINE_RE.match(line):
            sku = m["sku"]
        elif m := TITLE_FIELD_RE.match(line):
            name = m["name"]
        elif line != DISCONTINUED_MARK and not PRE_CATEGORY_RE.match(line):
            break
        start = j
        j -= 1
    if start - 1 >= floor:
        title = lines[start - 1][1]
        if not SECTION_RE.match(title) and not FIELD_RE.match(title):
            if not name:
                start -= 1
                if m := CODE_TITLE_RE.match(title):
                    sku = sku or m["sku"]
                    name = m["name"]
                else:
                    name = NUMBERED_RE.sub("", title)
            elif title == name:
                start -= 1  # "Product title:" layouts repeat the name on the line above
    if start - 1 >= floor and lines[start - 1][1] == DISCONTINUED_MARK:
        start -= 1
    return start, sku, name


def parse_catalog(pages: list[str], source: str = "") -> tuple[list[tuple[int, str]], list[Product]]:
    """Split a catalog into products and "notes" (any text that is not a product).

    Returns (notes, products) where notes is a list of (page, text) blocks: the cover page,
    warehouse legends, operating notes, summary tables...
    """
    lines = _lines_with_pages(pages)
    cat_idxs = [i for i, (_, line) in enumerate(lines) if CATEGORY_RE.match(line)]

    starts = []
    floor = 0
    for ci in cat_idxs:
        start, sku, name = _find_start(lines, ci, floor)
        starts.append((start, ci, sku, name))
        floor = ci + 1

    notes: list[tuple[int, str]] = []

    def add_note(chunk: list[tuple[int, str]]) -> None:
        if chunk:
            notes.append((chunk[0][0], "\n".join(line for _, line in chunk)))

    products = []
    section = ""
    pos = 0  # first line not yet assigned to a product or a note
    for n, (start, anchor, sku, name) in enumerate(starts):
        # Text before this product that no product claimed (headers, notes pages)
        gap = lines[pos:start]
        for _, line in gap:
            if m := SECTION_RE.match(line):
                section = m["section"]
        add_note([(pg, ln) for pg, ln in gap if not SECTION_RE.match(ln)])

        # A product ends at the next product or at the next catalog section header
        hard_end = starts[n + 1][0] if n + 1 < len(starts) else len(lines)
        end = hard_end
        for k in range(anchor + 1, hard_end):
            if SECTION_RE.match(lines[k][1]):
                end = k
                break
        body = [line for _, line in lines[start:end]]
        p = Product(sku=sku, name=name, section=section, page=lines[start][0], source=source)
        p.text = "\n".join(body)
        _fill_fields(p, body)
        if products and p.sku and p.sku == products[-1].sku and (not p.name or products[-1].price_eur is None):
            # Same product printed twice in a row (header repeated mid-record): merge
            prev = products[-1]
            prev.text += "\n" + p.text
            _fill_fields(prev, prev.text.splitlines())
        elif p.price_eur is not None or re.search(r"[\d-]", p.sku):
            products.append(p)
        else:  # e.g. a "Catalog control information" page that also has a Category: line
            add_note(lines[start:end])
        pos = end
    add_note([(pg, ln) for pg, ln in lines[pos:] if not SECTION_RE.match(ln)])
    return notes, products


def _fill_fields(p: Product, body: list[str]) -> None:
    for line in body:
        m = FIELD_RE.match(line)
        if not m:
            continue
        key, value = m["key"].strip().lower(), m["value"].strip()
        if key == "sku" and not p.sku:
            p.sku = value.split()[0] if value else ""
        elif key == "category" and not p.category:
            p.category = value
        elif key == "price" and not p.price_text:
            p.price_text = value
            if pm := PRICE_RE.search(value):
                p.price_eur = float(pm["amount"].replace(",", ""))
        elif key == "availability" and not p.availability:
            p.availability = value
        elif key == "warehouse location" and not p.warehouse:
            p.warehouse = value
        elif key == "replacement sku" and not p.replacement_sku:
            p.replacement_sku = value
    if DISCONTINUED_MARK in body[:1] and "discontinued" not in p.availability.lower():
        p.availability = f"Discontinued; {p.availability}".strip("; ")
