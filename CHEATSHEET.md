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

- **One table, many tenants:** add `namespace_column="namespace"` to `init_vectorstore_table`
  and `namespace="acme"` to the store.
- **Large tables:** `store.apply_vector_index(CSPANNIndex())` builds a C-SPANN vector index
  (`from langchain_cockroachdb import CSPANNIndex`).

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

## Retries

Every call retries CockroachDB's transient errors on its own. To also cut off
calls that hang, pass a policy:

```python
from langchain_cockroachdb import RetryPolicy

engine = CockroachDBEngine.from_connection_string(
    URL, retry_policy=RetryPolicy(attempt_timeout=10)
)
```
