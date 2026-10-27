# LangChain and CockroachDB: hackathon starter kit

Build an AI app on CockroachDB in an afternoon. This kit gives you a database
in one command, three working starting points, and a one-page
[cheat sheet](CHEATSHEET.md). It uses [langchain-cockroachdb](https://pypi.org/project/langchain-cockroachdb/),
which gives LangChain and LangGraph a vector store, hybrid search, chat
history, a checkpointer and a long-term memory store on CockroachDB.

## Agenda for 2 to 3 hours

| Time | What |
|---|---|
| 20 min | Intro: LangChain's building blocks (models, tools, agents) and where CockroachDB fits |
| 90 min | Build in pairs: pick a track below and make it yours |
| 20 min | Demos |

## Before the day

- Python 3.11 or newer.
- Docker Desktop or Podman, to run CockroachDB locally. A free CockroachDB Cloud cluster works too.
- An API key for one model provider. OpenAI is the default; see [Other model providers](#other-model-providers).
  No key yet? Set `OFFLINE=1` and everything runs with stand-in models.

## Setup, about 10 minutes

```bash
git clone https://github.com/viragtripathi/langchain-cockroachdb-hackathon.git
cd langchain-cockroachdb-hackathon
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then add your API key, or set OFFLINE=1
docker compose up -d --wait     # CockroachDB on localhost:26257, DB Console on localhost:8080
python check_setup.py           # tells you what to fix, if anything
```

On macOS and Linux, `make setup`, `make db` and `make check` do the same.

## The tracks

Each track runs as is. Every script prints the tool calls the agent makes, so
you can see what it does. Pass a question to ask it once, or leave it out to
chat in the terminal.

### Track 1: RAG chatbot

A support bot for Acme Outfitters, an outdoor gear store, answering from the
docs in `data/acme/`. The docs go into a CockroachDB vector store with hybrid
search, which ranks by meaning and by keywords at once, so product codes such
as "TX-200" match exactly.

```bash
python tracks/1_rag_chatbot.py "Can I return my boots?"
python tracks/1_rag_chatbot.py "How much does the TX-200 weigh?"
```

Make it yours: load your own documents, split long ones into chunks, add
conversation memory, or filter by category.

### Track 2: Agent with memory

An assistant that remembers. `CockroachDBSaver` keeps each conversation (a
thread), and `CockroachDBStore` keeps facts about each user across all of their
threads, with semantic search.

```bash
python tracks/2_agent_memory.py --user alice --thread trip-1 "Remember that I am vegetarian."
python tracks/2_agent_memory.py --user alice --thread trip-2 "What should I eat tonight?"
python tracks/2_agent_memory.py --user bob --thread trip-3 "What should I eat tonight?"
```

Make it yours: give memories an expiry, add a "forget me" tool, or merge it
with track 1 so the support bot remembers each customer.

### Track 3: Multi-tenant RAG

Two companies, Acme Outfitters and Globex Coffee Roasters, share one vector
table. Each company's docs live in their own namespace, so a search never
crosses tenants. Ask both the same question:

```bash
python tracks/3_multi_tenant_rag.py --tenant acme "What is your return policy?"
python tracks/3_multi_tenant_rag.py --tenant globex "What is your return policy?"
```

Make it yours: add a third company, pick the tenant from a login, or give each
tenant its own tone of voice.

### Stretch: keep chatting through a region outage

The [CockroachDB chaos demo](https://github.com/viragtripathi/cockroach-chaos-demo)
runs a multi-region cluster on your laptop, with a presenter console to take
nodes, zones or whole regions down.

1. Stop this kit's database (`docker compose down`), since both use port 26257,
   or start the kit's on another port (see Troubleshooting).
2. Start the chaos demo with `./run.sh up` in its folder, and point
   `COCKROACHDB_URL` in `.env` at it:
   `postgresql://root@localhost:26257/defaultdb?sslmode=disable`.
3. Chat with track 2 and take a region down from http://localhost:8088/presenter.
   The demo's default multi-region layout survives losing a whole region, so
   the agent keeps answering: CockroachDB keeps serving, and the connector
   retries the calls the failure interrupted.

## Other model providers

Set `MODEL` and `EMBEDDINGS` in `.env` and install the provider's package. The
chat model has to support tool calling, which most current models do.

| Provider | MODEL | EMBEDDINGS | Package |
|---|---|---|---|
| OpenAI | `openai:gpt-5.5` | `openai:text-embedding-3-small` | `langchain-openai` |
| Google Gemini | `google_genai:gemini-2.5-flash-lite` | `google_genai:models/gemini-embedding-001` | `langchain-google-genai` |
| AWS Bedrock | `bedrock_converse:<model id>` | `bedrock:amazon.titan-embed-text-v2:0` | `langchain-aws` |
| Ollama, local and free | `ollama:llama3.2` | `ollama:nomic-embed-text` | `langchain-ollama` |
| Anthropic | `anthropic:<model>` | another provider's embeddings | `langchain-anthropic` |

The kit measures the embedding size on its own, so any embedding model works.

Small local models run tracks 1 and 3 well, but they often write a tool call
out as text instead of making it, so track 2 may not save memories with them.
For track 2, use a hosted model or a larger local one.

## Offline mode

With `OFFLINE=1`, a scripted model stands in for the LLM: it calls one of the
agent's tools with your message and answers with what the tool returned. A
keyword-based embedding stands in for a real one, so texts that share words
come out as similar. It shows that the database, the tools and the agent are
wired up. It does not give smart answers, so switch to a real model to build.

## Troubleshooting

- **Port 26257 or 8080 is already in use.** Start the database on other ports, and
  change the port in `COCKROACHDB_URL` to match:
  `CRDB_PORT=26258 CRDB_UI_PORT=8081 docker compose up -d --wait`.
- **Start over with an empty database.** `docker compose down`, then `docker compose up -d --wait`.
- **CockroachDB Cloud.** Paste your cluster's connection string into `COCKROACHDB_URL`.
  Both `postgresql://` and `cockroachdb://` URLs work.
- **See your tables.** The DB Console is at http://localhost:8080.

## Learn more

- LangChain: [quickstart](https://docs.langchain.com/oss/python/langchain/quickstart),
  [build a semantic search engine](https://docs.langchain.com/oss/python/langchain/knowledge-base),
  [short-term memory](https://docs.langchain.com/oss/python/langchain/short-term-memory),
  [long-term memory](https://docs.langchain.com/oss/python/langchain/long-term-memory),
  [build a SQL agent](https://docs.langchain.com/oss/python/langchain/sql-agent)
- langchain-cockroachdb: [docs](https://cockroachdb.github.io/langchain-cockroachdb/),
  [examples](https://cockroachdb.github.io/langchain-cockroachdb/examples/),
  [source](https://github.com/cockroachdb/langchain-cockroachdb)

## Testing the kit

Tested on Python 3.11, 3.13 and 3.15 with CockroachDB v26.3.2, in offline mode,
with OpenAI's default models, and with local Ollama models.

The tests run every track the way a participant would, in offline mode,
against the database in `COCKROACHDB_URL`:

```bash
OFFLINE=1 pytest -q      # or: make test
```
