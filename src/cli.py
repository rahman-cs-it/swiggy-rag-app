"""
cli.py
------
Simple command-line interface for the Swiggy Annual Report RAG system.

Usage:
    python src/cli.py
Then type your questions at the prompt. Type 'exit' or 'quit' to stop.
"""

from rag import SwiggyRAG


def main():
    print("Loading Swiggy Annual Report RAG system ...")
    rag = SwiggyRAG()
    print("Ready. Ask a question about the Swiggy Annual Report (FY 2023-24).")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        result = rag.answer(query)

        print(f"\nAnswer:\n{result['answer']}\n")
        print("Supporting context:")
        for c in result["contexts"]:
            preview = c["text"][:200].replace("\n", " ")
            print(f"  - (Page {c['page_number']}, score={c['score']:.3f}) {preview}...")
        print()


if __name__ == "__main__":
    main()
