"""Settings, models and the database URL, shared by every track.

MODEL and EMBEDDINGS in .env pick any LangChain provider, written as
"provider:model". With OFFLINE=1 the tracks run without API keys: a scripted
model calls the agent's tools, and a keyword-based embedding stands in for a
real one. Offline mode shows that everything is wired up. It does not give
smart answers.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DEFAULT_URL = "cockroachdb://root@localhost:26257/defaultdb?sslmode=disable"
DEFAULT_MODEL = "openai:gpt-5.5"
DEFAULT_EMBEDDINGS = "openai:text-embedding-3-small"


def offline() -> bool:
    """True when OFFLINE is set, so no API key is needed."""
    return os.getenv("OFFLINE", "0").strip().lower() in {"1", "true", "yes"}


def db_url() -> str:
    """The CockroachDB connection string from .env."""
    return os.getenv("COCKROACHDB_URL", DEFAULT_URL)


def chat_model() -> BaseChatModel:
    """The chat model named in MODEL, or the offline stand-in."""
    if offline():
        return OfflineChatModel()
    from langchain.chat_models import init_chat_model

    return init_chat_model(os.getenv("MODEL", DEFAULT_MODEL))


def embeddings() -> Embeddings:
    """The embedding model named in EMBEDDINGS, or the offline stand-in."""
    if offline():
        return KeywordEmbeddings()
    from langchain.embeddings import init_embeddings

    return init_embeddings(os.getenv("EMBEDDINGS", DEFAULT_EMBEDDINGS))


def embedding_dims(model: Embeddings) -> int:
    """The vector size of an embedding model, measured with one call."""
    return len(model.embed_query("How long is this vector?"))


_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "what",
    "how",
    "are",
    "you",
    "your",
    "our",
    "this",
    "that",
    "from",
    "can",
    "does",
    "have",
    "about",
    "into",
    "when",
    "which",
    "will",
    "there",
    "they",
    "them",
    "its",
    "any",
    "all",
    "but",
    "not",
    "was",
    "were",
    "has",
    "had",
    "who",
    "why",
    "where",
    "tell",
    "please",
    "get",
    "much",
    "many",
    "long",
    "work",
    "sell",
    "need",
    "want",
    "know",
    "like",
    "make",
    "use",
    "give",
    "here",
    "more",
    "some",
    "also",
    "than",
    "then",
    "just",
    "only",
    "very",
    "would",
    "could",
    "should",
    "out",
    "one",
}


def _stem(word: str) -> str:
    """A rough stem, so that "shipping", "ships" and "ship" match."""
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"  # "policies" -> "policy"
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "aeiou":
        word = word[:-1]  # "shipp" -> "ship"
    return word


class KeywordEmbeddings(Embeddings):
    """Offline stand-in for an embedding model: texts that share words get similar vectors.

    Each distinct word stem sets one position in the vector. Real embedding
    models match by meaning instead, so they also find texts that use other words.
    """

    dims = 2048

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dims
        vector[0] = 0.01  # never all zeros, so cosine distance is always defined
        stems = {
            _stem(word)
            for word in re.findall(r"[a-z0-9]+", text.lower())
            if len(word) >= 3 and word not in _STOPWORDS
        }
        for stem in stems:
            bucket = int(hashlib.sha256(stem.encode()).hexdigest(), 16) % (
                self.dims - 1
            )
            vector[bucket + 1] = 1.0
        norm = math.sqrt(sum(x * x for x in vector))
        return [x / norm for x in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class OfflineChatModel(BaseChatModel):
    """Offline stand-in for an LLM.

    It calls one of the agent's tools with the user's message: save_memory when
    the message starts with "remember", otherwise the first other tool, with
    the message as its query. Then it answers with what the tool returned.
    """

    tool_names: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "offline"

    def bind_tools(self, tools: Any, **kwargs: Any) -> OfflineChatModel:
        names = [
            getattr(t, "name", None) or getattr(t, "__name__", str(t)) for t in tools
        ]
        return self.model_copy(update={"tool_names": names})

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        last = messages[-1]
        if isinstance(last, HumanMessage) and self.tool_names:
            text = str(last.content)
            if text.lower().startswith("remember") and "save_memory" in self.tool_names:
                name, args = "save_memory", {"text": text}
            else:
                others = [n for n in self.tool_names if n != "save_memory"]
                name, args = (others or self.tool_names)[0], {"query": text}
            call = {"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}"}
            message = AIMessage(content="", tool_calls=[call])
        elif isinstance(last, ToolMessage):
            message = AIMessage(content=f"(offline answer) {last.content}")
        else:
            message = AIMessage(content="(offline mode) There is no tool to call.")
        return ChatResult(generations=[ChatGeneration(message=message)])
