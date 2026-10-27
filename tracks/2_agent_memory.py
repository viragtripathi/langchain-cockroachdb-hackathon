"""Track 2: an agent that remembers.

Two kinds of memory, both kept in CockroachDB:
- CockroachDBSaver saves each conversation (a "thread"), so the agent knows
  what was said earlier in it.
- CockroachDBStore keeps facts about each user across all of their threads,
  with semantic search to find the ones that matter.

    python tracks/2_agent_memory.py --user alice --thread trip-1 "Remember that I am vegetarian."
    python tracks/2_agent_memory.py --user alice --thread trip-2 "What should I eat tonight?"
    python tracks/2_agent_memory.py --user alice --thread trip-2    # keep chatting in that thread

Make it yours:
- Give memories an expiry: pass ttl={"default_ttl": 60 * 24 * 30} to the store (minutes).
- Add a tool that forgets a user: store.delete_namespace(("users", user_id)).
- Combine it with track 1, so the support bot remembers each customer.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent)
)  # lets "kit" import from any folder

from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_cockroachdb import CockroachDBSaver, CockroachDBStore

from kit.chat import chat
from kit.common import chat_model, db_url, embedding_dims, embeddings

SYSTEM_PROMPT = (
    "You are a friendly assistant with a long-term memory. When the user tells you "
    "something about themselves that is worth keeping, save it with save_memory. "
    "Before answering a question about the user, look it up with recall."
)


@dataclass
class Context:
    """Who the agent is talking to. Passed with every call."""

    user_id: str


def memories(runtime: ToolRuntime[Context]) -> tuple[str, ...]:
    """The namespace for the current user's memories."""
    return ("users", runtime.context.user_id, "memories")


@tool
def save_memory(text: str, runtime: ToolRuntime[Context]) -> str:
    """Save a fact about the user, to remember in future conversations."""
    runtime.store.put(memories(runtime), str(uuid.uuid4()), {"text": text})
    return f"Saved: {text}"


@tool
def recall(query: str, runtime: ToolRuntime[Context]) -> str:
    """Look up facts saved about the user that match the query."""
    found = runtime.store.search(memories(runtime), query=query, limit=3)
    if not found:
        return "No memories yet."
    return "\n".join(f"- {item.value['text']}" for item in found)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--user", default="alice", help="who is talking (default: alice)"
    )
    parser.add_argument(
        "--thread",
        default="chat-1",
        help="the conversation to continue (default: chat-1)",
    )
    parser.add_argument(
        "message", nargs="*", help="one message to send; leave it out to chat"
    )
    args = parser.parse_args()

    embedding_model = embeddings()
    dims = embedding_dims(embedding_model)
    index = {"dims": dims, "embed": embedding_model, "fields": ["text"]}

    with (
        # One store table per vector size, so switching embedding models never clashes
        CockroachDBStore.from_conn_string(
            db_url(), index=index, table_name=f"memories_{dims}"
        ) as store,
        CockroachDBSaver.from_conn_string(db_url()) as checkpointer,
    ):
        store.setup()  # creates the tables and the vector index; safe to run every time
        checkpointer.setup()
        agent = create_agent(
            chat_model(),
            tools=[save_memory, recall],
            system_prompt=SYSTEM_PROMPT,
            context_schema=Context,
            checkpointer=checkpointer,
            store=store,
        )
        print(f"User {args.user}, thread {args.thread}")
        chat(
            agent,
            " ".join(args.message) or None,
            config={"configurable": {"thread_id": args.thread}},
            context=Context(user_id=args.user),
        )


if __name__ == "__main__":
    main()
