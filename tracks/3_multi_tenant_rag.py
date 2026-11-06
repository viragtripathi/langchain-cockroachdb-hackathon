"""Track 3: one table, many customers.

Acme Outfitters and Globex Coffee Roasters share one vector table. Each
company's docs live in their own namespace, and a store opened for one
namespace only reads and writes that namespace. Ask both the same question
and each gets its own answer.

The table has one C-SPANN vector index, with the namespace as its prefix
column. From langchain-cockroachdb 0.6.0, a tenant's search with a category
filter takes its nearest rows from that tenant's part of the index first, and
scans only when too few of them pass. Before, the filter always made the
search scan the table, which gets slower as tenants and docs pile up.

    python tracks/3_multi_tenant_rag.py --tenant acme "What is your return policy?"
    python tracks/3_multi_tenant_rag.py --tenant globex "What is your return policy?"
    python tracks/3_multi_tenant_rag.py --tenant globex     # chat in the terminal

Make it yours:
- Add a third company in data/ and in TENANTS, and any new category in Category.
- Pick the tenant from a login instead of a flag.
- Give each tenant its own system prompt or tone of voice.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent)
)  # lets "kit" import from any folder

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_cockroachdb import CockroachDBEngine, CockroachDBVectorStore, CSPANNIndex

from kit.chat import chat
from kit.common import chat_model, db_url, embedding_dims, embeddings
from kit.docs import load_docs

TABLE = "support_docs"
TENANTS = {
    "acme": "Acme Outfitters, an outdoor gear store",
    "globex": "Globex Coffee Roasters, a coffee subscription company",
}
# The categories in the docs' "category:" lines. As a Literal, they go into the
# tool's schema, so a model only picks one that exists
Category = Literal["guides", "policies", "products", "programs"]


def load_tenants(engine: CockroachDBEngine) -> dict[str, CockroachDBVectorStore]:
    """Create the shared table and load every tenant's docs into its own namespace."""
    embedding_model = embeddings()
    engine.init_vectorstore_table(
        table_name=TABLE,
        vector_dimension=embedding_dims(embedding_model),
        namespace_column="namespace",  # turns on namespaces for this table
        drop_if_exists=True,  # start fresh on every run
    )
    stores = {
        tenant: CockroachDBVectorStore(
            engine=engine,
            embeddings=embedding_model,
            collection_name=TABLE,
            namespace=tenant,
        )
        for tenant in TENANTS
    }
    # The index belongs to the table, so one tenant's store builds it for all,
    # while the table is empty. The namespace prefix keeps each tenant's
    # vectors together in the index
    first = next(iter(stores.values()))
    first.apply_vector_index(CSPANNIndex(), prefix_columns=["namespace"])
    for tenant, store in stores.items():
        store.add_documents(load_docs(tenant))
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
        def search_docs(query: str, category: Category | None = None) -> str:
            """Search this company's help docs.

            Args:
                query: What to look for, in plain words.
                category: Optional filter: only docs in this category.
            """
            found = store.similarity_search(
                query, k=2, filter={"category": category} if category else None
            )
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
