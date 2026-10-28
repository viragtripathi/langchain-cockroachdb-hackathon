"""Run every track the way a participant would, in offline mode.

Needs a running CockroachDB at COCKROACHDB_URL (docker compose up -d --wait):

    OFFLINE=1 pytest -q
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(*args: str) -> str:
    env = dict(os.environ, OFFLINE="1")
    done = subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,  # the assert below shows the output when a track fails
    )
    assert done.returncode == 0, f"{args} failed:\n{done.stdout}\n{done.stderr}"
    return done.stdout


def test_check_setup_passes() -> None:
    out = run("check_setup.py")
    assert "langchain-cockroachdb: OK" in out
    assert "Database: OK" in out
    assert "offline mode" in out


def test_check_setup_flags_an_old_langchain_cockroachdb() -> None:
    # Someone who installed the kit while it still asked for 0.5.0
    fake_old_version = (
        "import importlib.metadata as metadata, runpy\n"
        "real = metadata.version\n"
        "metadata.version = lambda name: (\n"
        "    '0.5.0' if name == 'langchain-cockroachdb' else real(name))\n"
        "runpy.run_path('check_setup.py', run_name='__main__')\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", fake_old_version],
        cwd=ROOT,
        env=dict(os.environ, OFFLINE="1"),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 1, done.stdout + done.stderr
    assert "langchain-cockroachdb: 0.5.0 is older than 0.5.1" in done.stdout
    assert "pip install -r requirements.txt" in done.stdout


def test_rag_chatbot_finds_the_return_policy() -> None:
    out = run("tracks/1_rag_chatbot.py", "How many days do I have to return an item?")
    assert "Loaded 8 docs" in out
    assert "[Returns]" in out and "30 days" in out


def test_rag_chatbot_matches_an_exact_product_code() -> None:
    out = run("tracks/1_rag_chatbot.py", "How much does the TX-200 weigh?")
    assert "[Trailhead TX-200 tent]" in out


def test_agent_memory_carries_across_threads_and_stays_per_user() -> None:
    user = f"user-{uuid.uuid4().hex[:6]}"
    saved = run(
        "tracks/2_agent_memory.py",
        "--user",
        user,
        "--thread",
        f"{user}-1",
        "Remember that I am vegetarian.",
    )
    assert "Saved: Remember that I am vegetarian." in saved
    recalled = run(
        "tracks/2_agent_memory.py",
        "--user",
        user,
        "--thread",
        f"{user}-2",
        "What do I like to eat?",
    )
    assert "vegetarian" in recalled
    stranger = run(
        "tracks/2_agent_memory.py",
        "--user",
        f"{user}-other",
        "--thread",
        f"{user}-3",
        "What do I like to eat?",
    )
    assert "No memories yet." in stranger


def test_each_tenant_only_sees_its_own_docs() -> None:
    acme = run(
        "tracks/3_multi_tenant_rag.py",
        "--tenant",
        "acme",
        "What is your return policy?",
    )
    globex = run(
        "tracks/3_multi_tenant_rag.py",
        "--tenant",
        "globex",
        "What is your return policy?",
    )
    assert "30 days" in acme and "opened bags" not in acme
    assert "opened bags" in globex and "30 days" not in globex
