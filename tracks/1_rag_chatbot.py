"""Track 1: a RAG chatbot that answers from Acme Outfitters' help docs.

The docs go into a CockroachDB vector store with hybrid search, which ranks by
meaning and by keywords at the same time, so exact product codes such as
"TX-200" match too. The agent searches the docs with a tool and answers from
what it finds.

    python tracks/1_rag_chatbot.py                         # chat in the terminal
    python tracks/1_rag_chatbot.py "Can I return my boots?"  # ask one question

Make it yours:
- Load your own documents into data/, or from a PDF or a web page.
- Split long documents into chunks before adding them (langchain-text-splitters).
- Let the agent remember the conversation: give create_agent a CockroachDBSaver (see track 2).
- Have the agent use the category filter, or return more results with a larger k.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent)
)  # lets "kit" import from any folder

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_cockroachdb import (
    CockroachDBEngine,
    CockroachDBVectorStore,
    HybridSearchConfig,
)

from kit.chat import chat
from kit.common import chat_model, db_url, embedding_dims, embeddings
from kit.docs import load_docs

TABLE = "acme_docs"

SYSTEM_PROMPT = (
    "You are the support assistant for Acme Outfitters, an outdoor gear store. "
    "Answer only from the docs that search_docs returns, and name the docs you used. "
    "If the docs do not answer the question, say so."
)


def build_vector_store(engine: CockroachDBEngine) -> CockroachDBVectorStore:
    """Create the table and load the docs. It runs on every start, so your edits to data/ show up."""
    embedding_model = embeddings()
    engine.init_vectorstore_table(
        table_name=TABLE,
        vector_dimension=embedding_dims(embedding_model),
        create_tsvector=True,  # adds full-text search, for hybrid search
        drop_if_exists=True,  # start fresh on every run
    )
    store = CockroachDBVectorStore(
        engine=engine,
        embeddings=embedding_model,
        collection_name=TABLE,
        hybrid_search_config=HybridSearchConfig(),
    )
    docs = load_docs("acme")
    store.add_documents(docs)
    print(f"Loaded {len(docs)} docs into the {TABLE} table.")
    return store


def main() -> None:
    engine = CockroachDBEngine.from_connection_string(db_url())
    try:
        store = build_vector_store(engine)

        @tool
        def search_docs(query: str, category: str | None = None) -> str:
            """Search Acme Outfitters' help docs.

            Args:
                query: What to look for, in plain words.
                category: Optional filter: "policies", "products" or "programs".
            """
            found = store.similarity_search(
                query, k=3, filter={"category": category} if category else None
            )
            if not found:
                return "No matching docs."
            return "\n\n".join(
                f"[{doc.metadata['title']}] {doc.page_content}" for doc in found
            )

        agent = create_agent(
            chat_model(), tools=[search_docs], system_prompt=SYSTEM_PROMPT
        )
        chat(agent, " ".join(sys.argv[1:]) or None)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
