"""One command to go from documents.zip to a ready-to-query index.

    python -m ironstore_assistant.build_index                 # OpenAI embeddings (needs OPENAI_API_KEY)
    python -m ironstore_assistant.build_index --embeddings tfidf   # offline, no key
"""

import argparse

from . import config
from .index import build_index
from .ingest import extract_zip, ingest, save_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings", choices=["openai", "huggingface", "tfidf"], default=config.EMBEDDINGS_PROVIDER)
    args = parser.parse_args()

    print(f"1/3 Extracting {config.DOCUMENTS_ZIP.name} (skipping __MACOSX / .DS_Store)…")
    paths = extract_zip()
    print(f"    {len(paths)} PDFs")
    print("2/3 Parsing, cleaning and chunking…")
    result = ingest(paths)
    save_outputs(result)
    print(f"    {len(result.documents)} chunks, {len(result.products)} products")
    print(f"    data-quality report: {config.QUALITY_REPORT}")
    print(f"3/3 Embedding with {args.embeddings} and saving FAISS index to {config.INDEX_DIR}…")
    build_index(result.documents, provider=args.embeddings)
    print("Done.")


if __name__ == "__main__":
    main()
