"""Track 3: one table, many customers.

Acme Outfitters and Globex Coffee Roasters share one vector table. Each
company's docs live in their own namespace, and a store opened for one
namespace only reads and writes that namespace. Ask both the same question
and each gets its own answer.

    python tracks/3_multi_tenant_rag.py --tenant acme "What is your return policy?"
    python tracks/3_multi_tenant_rag.py --tenant globex "What is your return policy?"
    python tracks/3_multi_tenant_rag.py --tenant globex     # chat in the terminal

Make it yours:
- Add a third company in data/ and in TENANTS.
- Pick the tenant from a login instead of a flag.
- Give each tenant its own system prompt or tone of voice.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent)
)  # lets "kit" import from any folder

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_cockroachdb import CockroachDBEngine, CockroachDBVectorStore

from kit.chat import chat
from kit.common import chat_model, db_url, embedding_dims, embeddings
from kit.docs import load_docs

TABLE = "support_docs"
TENANTS = {
    "acme": "Acme Outfitters, an outdoor gear store",
    "globex": "Globex Coffee Roasters, a coffee subscription company",
}


def load_tenants(engine: CockroachDBEngine) -> dict[str, CockroachDBVectorStore]:
    """Create the shared table and load every tenant's docs into its own namespace."""
    embedding_model = embeddings()
    engine.init_vectorstore_table(
        table_name=TABLE,
        vector_dimension=embedding_dims(embedding_model),
        namespace_column="namespace",  # turns on namespaces for this table
        drop_if_exists=True,  # start fresh on every run
    )
    stores = {}
    for tenant in TENANTS:
        store = CockroachDBVectorStore(
            engine=engine,
            embeddings=embedding_model,
            collection_name=TABLE,
            namespace=tenant,
        )
        store.add_documents(load_docs(tenant))
        stores[tenant] = store
    print(f"Loaded {', '.join(TENANTS)} into one {TABLE} table.")
    return stores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tenant", choices=sorted(TENANTS), default="acme")
    parser.add_argument(
        "message", nargs="*", help="one question to ask; leave it out to chat"
    )
    args = parser.parse_args()

    engine = CockroachDBEngine.from_connection_string(db_url())
    try:
        store = load_tenants(engine)[args.tenant]

        @tool
        def search_docs(query: str) -> str:
            """Search this company's help docs."""
            found = store.similarity_search(query, k=2)
            if not found:
                return "No matching docs."
            return "\n\n".join(
                f"[{doc.metadata['title']}] {doc.page_content}" for doc in found
            )

        agent = create_agent(
            chat_model(),
            tools=[search_docs],
            system_prompt=(
                f"You are the support assistant for {TENANTS[args.tenant]}. "
                "Answer only from the docs that search_docs returns."
            ),
        )
        print(f"Tenant: {args.tenant}")
        chat(agent, " ".join(args.message) or None)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
