"""Settings, models and the database URL, shared by every track.

MODEL and EMBEDDINGS in .env pick any LangChain provider, written as
"provider:model". With OFFLINE=1 the tracks run without API keys: a scripted
model picks the agent's tools by the words in your message, and a
keyword-based embedding stands in for a real one. Offline mode shows that
everything is wired up. It does not give smart answers.
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


# The topics the offline model saves facts under, picked by the words a fact
# uses. A fact with none of these words goes under its first word instead.
OFFLINE_TOPICS = {
    "diet": {"diet", "vegetarian", "vegan", "eat", "eats", "fish", "meat"},
    "home_city": {"city", "live", "lives", "moved"},
}
# Words that ask how a memory changed. The offline model calls memory_history for them
HISTORY_WORDS = ("changed", "history", "before", "used to")


def _topic(text: str) -> str | None:
    """The topic in OFFLINE_TOPICS whose words the text uses, if any."""
    words = set(re.findall(r"[a-z]+", text.lower()))
    return next((topic for topic, keys in OFFLINE_TOPICS.items() if words & keys), None)


class OfflineChatModel(BaseChatModel):
    """Offline stand-in for an LLM.

    It calls one of the agent's tools with the user's message, then answers
    with what the tool returned. A message that starts with "remember" goes to
    save_memory, with a topic from OFFLINE_TOPICS. A message that asks how
    something changed goes to memory_history. Any other message goes to the
    first other tool, as its query. If that tool takes a category and the
    message names one of the categories it allows, the category goes along too.
    """

    tool_names: list[str] = Field(default_factory=list)
    # For each tool that takes a category, the categories it allows
    categories: dict[str, list[str]] = Field(default_factory=dict)

    @property
    def _llm_type(self) -> str:
        return "offline"

    def bind_tools(self, tools: Any, **kwargs: Any) -> OfflineChatModel:
        names = [
            getattr(t, "name", None) or getattr(t, "__name__", str(t)) for t in tools
        ]
        categories = {}
        for name, t in zip(names, tools, strict=True):
            schema = (getattr(t, "args", None) or {}).get("category")
            if schema is not None:
                # A Literal lists its values in the schema. Otherwise take the
                # values in quotes in the description, as track 1 writes them
                options = [schema, *schema.get("anyOf", [])]
                allowed = [value for o in options for value in o.get("enum", [])]
                categories[name] = allowed or re.findall(
                    r'"([a-z_]+)"', getattr(t, "description", "")
                )
        return self.model_copy(update={"tool_names": names, "categories": categories})

    def _tool_call(self, text: str) -> tuple[str, dict[str, Any]]:
        """The tool to call for a message, and its arguments."""
        lowered = text.lower()
        if lowered.startswith("remember") and "save_memory" in self.tool_names:
            fact = re.sub(r"^remember(\s+that)?\W*", "", text, flags=re.IGNORECASE)
            fact = (fact[:1].upper() + fact[1:]) or text
            words = [
                word
                for word in re.findall(r"[a-z]+", fact.lower())
                if len(word) >= 3 and word not in _STOPWORDS
            ]
            topic = _topic(fact) or (words[0] if words else "notes")
            return "save_memory", {"text": fact, "topic": topic}
        if "memory_history" in self.tool_names and any(
            word in lowered for word in HISTORY_WORDS
        ):
            topic = _topic(text)
            return "memory_history", {"topic": topic} if topic else {}
        others = [
            n for n in self.tool_names if n not in ("save_memory", "memory_history")
        ]
        name = (others or self.tool_names)[0]
        args: dict[str, Any] = {"query": text}
        stems = {_stem(word) for word in re.findall(r"[a-z0-9]+", lowered)}
        named = [c for c in self.categories.get(name, []) if _stem(c) in stems]
        if named:
            args["category"] = named[0]
        return name, args

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        last = messages[-1]
        if isinstance(last, HumanMessage) and self.tool_names:
            name, args = self._tool_call(str(last.content))
            call = {"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}"}
            message = AIMessage(content="", tool_calls=[call])
        elif isinstance(last, ToolMessage):
            message = AIMessage(content=f"(offline answer) {last.content}")
        else:
            message = AIMessage(content="(offline mode) There is no tool to call.")
        return ChatResult(generations=[ChatGeneration(message=message)])
