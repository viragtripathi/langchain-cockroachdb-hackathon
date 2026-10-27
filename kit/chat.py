"""Run an agent on one question, or as a chat in the terminal."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


def ask(
    agent: Any, text: str, *, config: dict | None = None, context: Any = None
) -> str:
    """Send one message, print the tools the agent used, and return its answer."""
    result = agent.invoke(
        {"messages": [{"role": "user", "content": text}]},
        config=config,
        context=context,
    )
    messages = result["messages"]
    # With a checkpointer the result holds the whole thread, so show this turn only
    start = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    for message in messages[start + 1 :]:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
                print(f"   tool call: {call['name']}({args})")
        elif isinstance(message, ToolMessage):
            print(f"   tool result: {message.text[:300]}")
    return messages[-1].text


def chat(
    agent: Any,
    question: str | None = None,
    *,
    config: dict | None = None,
    context: Any = None,
) -> None:
    """Answer one question, or chat until you type quit."""
    if question:
        print(f"you> {question}")
        print(f"bot> {ask(agent, question, config=config, context=context)}")
        return
    print("Ask a question, or type quit to stop.")
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if text.lower() in {"quit", "exit", "q"}:
            return
        if text:
            print(f"bot> {ask(agent, text, config=config, context=context)}")
