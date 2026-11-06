"""Run every track the way a participant would, in offline mode.

Needs a running CockroachDB at COCKROACHDB_URL (docker compose up -d --wait):

    OFFLINE=1 pytest -q
"""

from __future__ import annotations

import os
import re
import runpy
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
    # Someone who installed the kit while it still asked for 0.5.1
    fake_old_version = (
        "import importlib.metadata as metadata, runpy\n"
        "real = metadata.version\n"
        "metadata.version = lambda name: (\n"
        "    '0.5.1' if name == 'langchain-cockroachdb' else real(name))\n"
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
    assert "langchain-cockroachdb: 0.5.1 is older than 0.6.0" in done.stdout
    assert "pip install -r requirements.txt" in done.stdout


def test_rag_chatbot_finds_the_return_policy() -> None:
    out = run("tracks/1_rag_chatbot.py", "How many days do I have to return an item?")
    assert "Loaded 8 docs" in out
    assert "[Returns]" in out and "30 days" in out


def test_rag_chatbot_matches_an_exact_product_code() -> None:
    out = run("tracks/1_rag_chatbot.py", "How much does the TX-200 weigh?")
    assert "[Trailhead TX-200 tent]" in out


def memory(user: str, *args: str) -> str:
    """Run track 2 as one user."""
    return run("tracks/2_agent_memory.py", "--user", user, *args)


def test_agent_memory_carries_across_threads_and_stays_per_user() -> None:
    user = f"user-{uuid.uuid4().hex[:6]}"
    saved = memory(user, "--thread", f"{user}-1", "Remember that I am vegetarian.")
    assert "save_memory(text='I am vegetarian.', topic='diet')" in saved
    assert "Saved (diet): I am vegetarian." in saved
    recalled = memory(user, "--thread", f"{user}-2", "What do I like to eat?")
    assert "diet: I am vegetarian." in recalled
    stranger = memory(
        f"{user}-other", "--thread", f"{user}-3", "What do I like to eat?"
    )
    assert "No memories yet." in stranger
    assert "No memories yet." in memory(f"{user}-other", "--history")


def test_agent_memory_spells_each_topic_one_way() -> None:
    track = runpy.run_path(str(ROOT / "tracks" / "2_agent_memory.py"))
    normalize_topic = track["normalize_topic"]
    assert normalize_topic("Home city") == normalize_topic("home_city") == "home_city"
    assert normalize_topic(" Diet ") == "diet"
    assert normalize_topic("") is None and normalize_topic(None) is None


def test_agent_memory_replaces_a_changed_fact_and_keeps_its_history() -> None:
    user = f"user-{uuid.uuid4().hex[:6]}"
    memory(user, "--thread", f"{user}-1", "Remember that I am vegetarian.")
    changed = memory(user, "--thread", f"{user}-2", "Remember that I eat fish now.")
    assert "Saved (diet): I eat fish now. It replaces: I am vegetarian." in changed

    # Recall finds the current fact only
    recalled = memory(user, "--thread", f"{user}-3", "What should I eat tonight?")
    assert "diet: I eat fish now." in recalled
    assert "vegetarian" not in recalled

    # Saying it again confirms the memory instead of adding a copy
    repeated = memory(user, "--thread", f"{user}-3", "Remember that I eat fish now.")
    assert "Already known (diet), so I confirmed it: I eat fish now." in repeated

    # The agent can look at the history. --history prints it without the agent,
    # though it still calls the embedding model once, to measure the vector size
    asked = memory(user, "--thread", f"{user}-3", "How has my diet changed?")
    assert "memory_history(topic='diet')" in asked
    assert "before: I am vegetarian." in asked
    history = memory(user, "--history")
    assert f"now: I eat fish now. (said in {user}-2 at " in history
    assert "confirmed 1 time)" in history
    assert f"before: I am vegetarian. (said in {user}-1 at " in history
    assert "; superseded at " in history

    # A topic with nothing on file says so, and names the topics there are
    other = memory(user, "--thread", f"{user}-3", "How has my home city changed?")
    assert "No memories about home_city. Topics on file: diet." in other


def test_agent_memory_keeps_topics_apart() -> None:
    user = f"user-{uuid.uuid4().hex[:6]}"
    memory(user, "--thread", f"{user}-1", "Remember that I am vegetarian.")
    memory(user, "--thread", f"{user}-1", "Remember that I live in Lisbon.")
    moved = memory(user, "--thread", f"{user}-2", "Remember that I moved to Porto.")
    assert (
        "Saved (home_city): I moved to Porto. It replaces: I live in Lisbon." in moved
    )
    # The home_city change leaves the diet fact alone
    diet, home_city = memory(user, "--history").split("home_city:")
    assert "now: I am vegetarian." in diet and "before:" not in diet
    assert "now: I moved to Porto." in home_city
    assert "before: I live in Lisbon." in home_city


def test_each_tenant_only_sees_its_own_docs_with_a_filter() -> None:
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
    # The question names a category, so the search filters on it
    assert "category='policies'" in acme and "category='policies'" in globex
    assert set(re.findall(r"\[([^\]]+)\]", acme)) <= {"Returns", "Shipping", "Warranty"}
    assert set(re.findall(r"\[([^\]]+)\]", globex)) <= {"Returns", "Shipping"}
