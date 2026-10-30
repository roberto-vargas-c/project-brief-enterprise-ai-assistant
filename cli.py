"""Terminal chat with the IronStore Assistant.

    python cli.py            # standard RAG
    python cli.py --agent    # tool-calling agent
"""

import argparse

from ironstore_assistant.agent import AgentAssistant
from ironstore_assistant.rag import build_assistant


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", action="store_true", help="use the tool-calling agent")
    parser.add_argument("--no-multi-query", action="store_true")
    args = parser.parse_args()

    rag = build_assistant(multi_query=not args.no_multi_query)
    bot = AgentAssistant(rag) if args.agent else rag
    history: list[tuple[str, str]] = []
    print("IronStore Assistant. Ask a question (empty line or Ctrl+C to quit).\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        resp = bot.ask(q, history=history)
        print(f"\nAssistant ({resp.confidence} confidence):\n{resp.answer}\n")
        for s in resp.sources:
            print(f"  [{s.number}] {s.label()} — {s.source}")
        for w in resp.warnings:
            print(f"  ⚠ {w}")
        print()
        history += [("user", q), ("assistant", resp.answer)]


if __name__ == "__main__":
    main()
