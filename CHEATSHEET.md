# Cheat sheet: LangChain with CockroachDB

```bash
pip install langchain-cockroachdb langchain langchain-openai
```

Every snippet below uses `URL`, your connection string, for example
`cockroachdb://root@localhost:26257/defaultdb?sslmode=disable`.

## Models

```python
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings

model = init_chat_model("openai:gpt-5.5")  # any "provider:model"
embeddings = init_embeddings("openai:text-embedding-3-small")  # 1536 dimensions
```

## Vector store, for RAG

```python
from langchain_cockroachdb import (
    CockroachDBEngine,
    CockroachDBVectorStore,
    HybridSearchConfig,
)

engine = CockroachDBEngine.from_connection_string(URL)
engine.init_vectorstore_table(
    table_name="docs", vector_dimension=1536, create_tsvector=True
)
store = CockroachDBVectorStore(
    engine=engine,
    embeddings=embeddings,
    collection_name="docs",
    hybrid_search_config=HybridSearchConfig(),  # keywords + meaning; leave out for vector only
)
store.add_texts(
    ["CockroachDB is a distributed SQL database."], metadatas=[{"topic": "db"}]
)
docs = store.similarity_search("What is CockroachDB?", k=3, filter={"topic": "db"})
retriever = store.as_retriever(search_kwargs={"k": 3})
```

- **Large tables:** `store.apply_vector_index(CSPANNIndex())` builds a C-SPANN vector index
  (`from langchain_cockroachdb import CSPANNIndex`). Build it before you add rows: on a table
  that holds data, the build blocks writes until it ends.

## One table, many tenants

```python
from langchain_cockroachdb import CSPANNIndex

engine.init_vectorstore_table(
    table_name="support", vector_dimension=1536, namespace_column="namespace"
)
acme = CockroachDBVectorStore(
    engine=engine, embeddings=embeddings, collection_name="support", namespace="acme"
)
# The namespace as the index's prefix column: a tenant reads only its own part
acme.apply_vector_index(CSPANNIndex(), prefix_columns=["namespace"])
docs = acme.similarity_search("return policy", k=3, filter={"category": "policies"})
```

Every row this store writes gets the namespace `acme`, and it reads only those
rows. New in 0.6.0: a search with a `filter` takes its nearest rows from the
index first, and scans only when too few pass. Track 3 shows it.

## Chat history

```python
from langchain_cockroachdb import CockroachDBChatMessageHistory

history = CockroachDBChatMessageHistory(
    session_id="session-1", engine=engine, table_name="chat_history"
)
history.create_table_if_not_exists()
history.add_user_message("Hi!")
history.add_ai_message("Hello! How can I help?")
print(history.messages)
```

## An agent with a tool

```python
from langchain.agents import create_agent
from langchain.tools import tool


@tool
def search_docs(query: str) -> str:
    """Search the docs."""
    return "\n\n".join(doc.page_content for doc in store.similarity_search(query, k=3))


agent = create_agent(model, tools=[search_docs], system_prompt="Answer from the docs.")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is CockroachDB?"}]}
)
print(result["messages"][-1].text)
```

## Conversation memory: the checkpointer

```python
from langchain_cockroachdb import CockroachDBSaver

with CockroachDBSaver.from_conn_string(URL) as checkpointer:
    checkpointer.setup()  # once; safe to repeat
    agent = create_agent(model, tools=[search_docs], checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "conversation-1"}}
    agent.invoke({"messages": [{"role": "user", "content": "Hi, I'm Sam."}]}, config)
    agent.invoke({"messages": [{"role": "user", "content": "What's my name?"}]}, config)
```

## Long-term memory: the store

```python
from langchain_cockroachdb import CockroachDBStore

index = {"dims": 1536, "embed": embeddings, "fields": ["text"]}
with CockroachDBStore.from_conn_string(URL, index=index) as memory:
    memory.setup()
    memory.put(("users", "sam"), "food", {"text": "Sam is vegetarian."})
    hits = memory.search(("users", "sam"), query="What does Sam eat?", limit=3)
    memory.delete_namespace(("users", "sam"))  # forget a user
```

Pass the store to `create_agent(..., store=memory)`. Inside a tool, take
`runtime: ToolRuntime[Context]` (from `langchain.tools`) and use `runtime.store`.
Track 2 shows the whole pattern.

## Memories that stay true (new in 0.6.0)

```python
from langchain_cockroachdb.store import first_of, skip_duplicates, supersede_same_slot

# A repeated fact confirms the saved one. Otherwise a newer fact in the same slot wins
policy = first_of(skip_duplicates(), supersede_same_slot())
sam = ("users", "sam")
facts = [("Sam is vegetarian.", "chat-1"), ("Sam eats fish now.", "chat-2")]
with CockroachDBStore.from_conn_string(URL, index=index, history=True) as memory:
    memory.setup()
    for text, thread in facts:
        result = memory.reconcile(
            sam,
            None,  # a new key
            {"topic": "diet", "text": text},
            slot="topic",  # memories with the same topic state the same fact
            policy=policy,
            provenance={"thread": thread},
        )
    # The version it replaced: what it said, why it ended, and where it came from
    old = memory.history(sam, result.superseded[0])[0]
    print(old.value["text"], old.reason, old.provenance)
    memory.confirm(sam, result.key)  # still true: count it and renew its expiry
```

Now a search finds only the newer fact. Without a key, `memory.history(sam)`
lists the old versions of every memory of Sam's, newest first, each with its
own `key`. To change one memory from two places without losing an update,
write it back only if it is still the one you read:

```python
with CockroachDBStore.from_conn_string(URL, index=index, history=True) as memory:
    record = memory.record(sam, "airlines")  # None if there is none yet
    while True:
        names = record.item.value["airlines"] if record else []
        airlines = sorted({*names, "KLM"})
        value = {"text": "Flies with " + ", ".join(airlines), "airlines": airlines}
        result = memory.put_if(sam, "airlines", value, expected=record)
        if result.applied:
            break
        record = result.current  # someone wrote it since you read it: merge with it
```

With history on, `delete` keeps the old versions. `memory.purge(sam, key)`
forgets one key with its history, but a fact that a newer one replaced keeps
its own key in history. So to forget a topic, purge every key that `search`
and `history(sam)` give for it. To forget all of Sam's memories, use
`memory.delete_namespace(sam)`.

## Retries

Every call retries CockroachDB's transient errors on its own. To also cut off
calls that hang, pass a policy:

```python
from langchain_cockroachdb import RetryPolicy

engine = CockroachDBEngine.from_connection_string(
    URL, retry_policy=RetryPolicy(attempt_timeout=10)
)
```
