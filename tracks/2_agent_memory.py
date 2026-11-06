"""Track 2: an agent that remembers, and keeps its memories true.

Two kinds of memory, both kept in CockroachDB:
- CockroachDBSaver saves each conversation (a "thread"), so the agent knows
  what was said earlier in it.
- CockroachDBStore keeps facts about each user across all of their threads,
  with semantic search to find the ones that matter.

New in langchain-cockroachdb 0.6.0, the store keeps those facts true. The
agent saves each fact under a short topic, such as "diet". A newer fact in the
same topic supersedes the older one, and a fact the user repeats confirms the
one already saved instead of adding a copy. So recall finds only current
facts. History keeps every old version with the thread it came from, so you
can see what the agent used to believe, when that changed, and in which
conversation.

    python tracks/2_agent_memory.py --user alice --thread trip-1 "Remember that I am vegetarian."
    python tracks/2_agent_memory.py --user alice --thread trip-2 "Remember that I eat fish now."
    python tracks/2_agent_memory.py --user alice --thread trip-3 "What should I eat tonight?"
    python tracks/2_agent_memory.py --user alice --thread trip-3 "How has my diet changed?"
    python tracks/2_agent_memory.py --user alice --history     # memories and old versions
    python tracks/2_agent_memory.py --user alice --thread trip-3    # keep chatting in that thread

Make it yours:
- Let memories expire unless the user confirms them: pass
  ttl={"default_ttl": 60 * 24 * 30, "refresh_on_read": False} to the store
  (minutes). A repeated fact renews the memory it repeats, and
  store.confirm(namespace, key) renews one by hand.
- Let two agents update one memory without losing a change: read it with
  store.record(namespace, key), write it with
  store.put_if(namespace, key, value, expected=record), and merge with
  result.current when result.applied is False.
- Add a "forget me" tool: store.delete_namespace(("users", user_id, "memories"))
  deletes a user's memories with their history. store.purge(namespace, key)
  forgets one key, and a fact that a newer one replaced keeps its own key in
  history, so to forget a topic, purge every key that store.search() and
  store.history(namespace) give for it. store.delete() keeps the history.
- Combine it with track 1, so the support bot remembers each customer.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent)
)  # lets "kit" import from any folder

from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_cockroachdb import CockroachDBSaver, CockroachDBStore
from langchain_cockroachdb.store import first_of, skip_duplicates, supersede_same_slot

from kit.chat import chat
from kit.common import chat_model, db_url, embedding_dims, embeddings

SYSTEM_PROMPT = (
    "You are a friendly assistant with a long-term memory. When the user tells you "
    "something about themselves that is worth keeping, save it with save_memory, "
    "under a short topic. Before saving, call recall and reuse the topic of a fact "
    "about the same thing, so the new fact replaces the old one. Before answering a "
    "question about the user, look it up with recall. When the user asks how "
    "something changed, or what they said before, use memory_history."
)

# A fact the user repeats is skipped, and confirms the memory already saved.
# Otherwise a newer fact in the same topic supersedes the older one
POLICY = first_of(skip_duplicates(), supersede_same_slot())


@dataclass
class Context:
    """Who the agent is talking to. Passed with every call."""

    user_id: str


def memories(user_id: str) -> tuple[str, ...]:
    """The namespace for a user's memories."""
    return ("users", user_id, "memories")


def normalize_topic(topic: str | None) -> str | None:
    """One spelling per topic, so "Home city" and "home_city" match. Empty is None."""
    return "_".join((topic or "").lower().split()) or None


@tool
def save_memory(text: str, topic: str, runtime: ToolRuntime[Context]) -> str:
    """Save a fact about the user, to remember in future conversations.

    Args:
        text: The fact, as a short sentence.
        topic: What the fact is about, in a word or two, such as "diet" or
            "home_city". Use the same topic for a fact that changes an earlier
            one, so the new fact replaces the old.
    """
    namespace = memories(runtime.context.user_id)
    topic = normalize_topic(topic)
    if topic is None:
        return "Give the fact a topic, such as diet or home_city, and save it again."
    result = runtime.store.reconcile(
        namespace,
        None,  # a new key
        {"topic": topic, "text": text},
        slot="topic",  # memories with the same topic state the same fact
        policy=POLICY,
        provenance={"thread": runtime.config["configurable"]["thread_id"]},
    )
    if result.new != "written":
        return f"Already known ({topic}), so I confirmed it: {text}"
    replaced = [
        version.value["text"]
        for key in result.superseded
        for version in runtime.store.history(namespace, key, limit=1)
    ]
    return f"Saved ({topic}): {text}" + (
        f" It replaces: {' '.join(replaced)}" if replaced else ""
    )


@tool
def recall(query: str, runtime: ToolRuntime[Context]) -> str:
    """Look up facts saved about the user that match the query."""
    found = runtime.store.search(
        memories(runtime.context.user_id), query=query, limit=3
    )
    if not found:
        return "No memories yet."
    return "\n".join(
        f"- {item.value.get('topic', 'no topic')}: {item.value['text']}"
        for item in found
    )


@tool
def memory_history(runtime: ToolRuntime[Context], topic: str | None = None) -> str:
    """Show how the user's facts changed over time: for each topic, the
    current fact, the older facts it replaced, and the thread each came from.

    Args:
        topic: Only this topic, such as "diet". Leave it out to see all.
    """
    return timeline(runtime.store, runtime.context.user_id, normalize_topic(topic))


def said(
    provenance: dict[str, Any] | None, at: datetime, confirmations: int = 0
) -> str:
    """Where and when a fact was said, and how often the user confirmed it."""
    thread = (provenance or {}).get("thread", "an unknown thread")
    line = f"said in {thread} at {at.astimezone(UTC):%Y-%m-%d %H:%M} UTC"
    if confirmations:
        line += f", confirmed {confirmations} time" + ("s" if confirmations > 1 else "")
    return line


def timeline(store: CockroachDBStore, user_id: str, topic: str | None = None) -> str:
    """A user's memories by topic: the current fact, then the older ones, newest first."""
    namespace = memories(user_id)
    topics: dict[str, list[str]] = {}
    for item in store.search(namespace, limit=100):
        record = store.record(namespace, item.key)  # with its provenance
        if record:
            topics.setdefault(item.value.get("topic", "no topic"), []).append(
                f"  now: {item.value['text']} "
                f"({said(record.provenance, record.valid_at, record.confirmations)})"
            )
    # Without a key, history() lists the old versions of every memory in the
    # namespace, newest first. Each new fact has a key of its own, so this is
    # how to find the facts that newer ones replaced
    for version in store.history(namespace, limit=100):
        ended = version.invalid_at or version.ended_at
        topics.setdefault(version.value.get("topic", "no topic"), []).append(
            f"  before: {version.value['text']} "
            f"({said(version.provenance, version.valid_at, version.confirmations)}; "
            f"{version.reason} at {ended.astimezone(UTC):%Y-%m-%d %H:%M} UTC)"
        )
    if not topics:
        return "No memories yet."
    if topic is not None:
        if topic not in topics:
            on_file = ", ".join(sorted(topics))
            return f"No memories about {topic}. Topics on file: {on_file}."
        topics = {topic: topics[topic]}
    return "\n".join(
        f"{name}:\n" + "\n".join(lines) for name, lines in sorted(topics.items())
    )


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
        "--history",
        action="store_true",
        help="print the user's memories with their old versions, then stop",
    )
    parser.add_argument(
        "message", nargs="*", help="one message to send; leave it out to chat"
    )
    args = parser.parse_args()

    embedding_model = embeddings()
    dims = embedding_dims(embedding_model)
    index = {"dims": dims, "embed": embedding_model, "fields": ["text"]}

    # One store table per vector size, so switching embedding models never clashes
    with CockroachDBStore.from_conn_string(
        db_url(), index=index, history=True, table_name=f"memories_{dims}"
    ) as store:
        store.setup()  # creates the tables and the vector index; safe to run every time
        if args.history:
            print(f"Memories of {args.user}:")
            print(timeline(store, args.user))
            return

        with CockroachDBSaver.from_conn_string(db_url()) as checkpointer:
            checkpointer.setup()
            agent = create_agent(
                chat_model(),
                tools=[save_memory, recall, memory_history],
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
