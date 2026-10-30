from datetime import date

from ironstore_assistant.catalog import parse_catalog
from ironstore_assistant.ingest import is_junk, parse_month_year


def test_junk_files_are_skipped(pdf_paths):
    assert len(pdf_paths) == 83
    assert not any("__MACOSX" in str(p) or p.name.startswith(".") for p in pdf_paths)
    assert is_junk("documents/.DS_Store")
    assert is_junk("__MACOSX/documents/._IronStore_Sales_Playbook.pdf")
    assert not is_junk("documents/internal_docs_by_area/Sales/IronStore_Sales_Playbook.pdf")


def test_parse_month_year():
    assert parse_month_year("August 2026") == date(2026, 8, 1)
    assert parse_month_year("1 September 2026") == date(2026, 9, 1)
    assert parse_month_year("n/a") is None


def test_inconsistent_headers_are_normalised(corpus):
    policies = {s.source: s for s in corpus.sources if s.doc_type == "policy"}
    assert len(policies) == 20
    # "Policy ID", "Document ID" and "Policy reference" all end up in doc_id
    assert policies["IronStore_Remote_Work_Policy.pdf"].meta["doc_id"] == "HR-REM-001"
    assert policies["IronStore_IT_Security_Policy.pdf"].meta["doc_id"] == "ITS-POL-001"
    assert policies["IronStore_Payroll_Processing_Procedures.pdf"].meta["doc_id"] == "FIN-PAY-004"
    # "Last reviewed in August 2026" is parsed too
    assert policies["IronStore_Leave_and_Absence_Policy.pdf"].meta["last_reviewed_date"] == "2026-08-01"
    # Area comes from the folder, not from the inconsistent "Document owner" field
    assert policies["IronStore_Remote_Work_Policy.pdf"].area == "HR policies"


def test_traps_are_detected(corpus):
    issues = corpus.issues
    assert any("CS-POL-014" in i for i in issues["Document ID shared by different documents"])
    outdated = [s.source for s in corpus.sources if s.meta["status"] == "outdated"]
    assert outdated == ["IronStore_Old_Product_Catalog_2021.pdf"]
    assert any(i.startswith("IS-BED-001") for i in issues["SKU reused by different products"])


def test_chunks_have_citation_metadata(corpus):
    assert len(corpus.documents) > 3000
    for d in corpus.documents:
        assert d.metadata["source"].endswith(".pdf")
        assert d.metadata["page"] >= 1
        assert d.metadata["title"]


def test_products_are_parsed_from_every_catalog(corpus):
    products = corpus.products
    assert products.source.nunique() == 63
    assert len(products) > 3000
    sofa = products[products.sku == "FUR-SOF-NOR-LYSA-GRY"].iloc[0]
    assert sofa["name"] == "Nordholm Lysa 3-Seater Sofa"
    assert sofa["price_eur"] == 899.0
    assert sofa["page"] == 1
    assert sofa["availability"].startswith("In stock")


def test_catalog_layouts():
    pages = [
        "IronStore Product Catalog\nDemo\nPage 1 — Things\n"
        "1. Alpha Lamp\nSKU: AA-001\nCategory: Demo > Lamps\nPrice: €10.00\nAvailability: In stock\n"
        "Warehouse location: NL\n"
        "BB-002 — Beta Chair\nCategory: Demo > Chairs\nPrice: €1,299.50\nAvailability: Low stock\n",
        "Warehouse location: DE\nPage 2 — Discontinued\nThis product has been discontinued\n"
        "Product title: Gamma Rug\nSKU: CC-003\nCategory: Demo > Rugs\nPrice: €5 at final sale\n"
        "Availability: Discontinued\nReplacement SKU: AA-001\nPage 3 — Notes\nSome closing notes.",
    ]
    notes, products = parse_catalog(pages, "demo.pdf")
    assert [(p.sku, p.name, p.price_eur, p.page) for p in products] == [
        ("AA-001", "Alpha Lamp", 10.0, 1),
        ("BB-002", "Beta Chair", 1299.5, 1),
        ("CC-003", "Gamma Rug", 5.0, 2),
    ]
    assert products[1].warehouse == "DE"  # field on the next PDF page still belongs to Beta
    assert products[2].replacement_sku == "AA-001"
    assert products[2].section == "Discontinued"
    assert any("closing notes" in text for _, text in notes)


def test_zip_slip_and_junk_are_ignored(tmp_path):
    import zipfile

    from ironstore_assistant.ingest import extract_zip

    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("../escape.pdf", b"%PDF-1.4")
        zf.writestr("__MACOSX/docs/._a.pdf", b"x")
        zf.writestr("docs/.DS_Store", b"x")
        zf.writestr("docs/ok.pdf", b"%PDF-1.4")
    out = extract_zip(z, tmp_path / "out")
    assert [p.name for p in out] == ["ok.pdf"]
    assert not (tmp_path / "escape.pdf").exists()


def test_chunks_keep_sentences_that_cross_page_breaks():
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from ironstore_assistant.ingest import split_across_pages

    pages = ["Intro.\n" + "filler " * 120 + "\nGoods are accepted within 30 calendar days of",
             "delivery. Exceptions apply.\n" + "more " * 50]
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50, add_start_index=True)
    chunks = split_across_pages(pages, splitter)
    spanning = [(first, last) for text, first, last in chunks if "30 calendar days of\ndelivery" in text]
    assert spanning == [(1, 2)]
    assert chunks[0][1] == 1 and chunks[-1][2] == 2
