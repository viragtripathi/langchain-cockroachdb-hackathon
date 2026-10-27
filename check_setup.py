"""Check that everything is ready: python check_setup.py

It checks your Python version, the database connection, and the chat and
embedding models in .env (or offline mode), and tells you what to fix.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    ok = True

    if sys.version_info < (3, 11):  # noqa: UP036 (tells people on older Python what to do)
        print(f"Python: {sys.version.split()[0]} is too old. Use Python 3.11 or newer.")
        return 1
    print(f"Python: OK ({sys.version.split()[0]})")

    try:
        from langchain_cockroachdb import CockroachDBSaver

        from kit.common import (
            DEFAULT_EMBEDDINGS,
            DEFAULT_MODEL,
            chat_model,
            db_url,
            embedding_dims,
            embeddings,
            offline,
        )
    except ImportError as error:
        print(f"Packages: missing ({error}). Run: pip install -r requirements.txt")
        return 1

    try:
        with CockroachDBSaver.from_conn_string(db_url()) as saver:
            version = saver.conn.execute("SELECT version()").fetchone()["version"]
        print(f"Database: OK ({version.split(' (')[0]})")
    except Exception as error:  # noqa: BLE001
        print(
            f"Database: cannot connect ({type(error).__name__}: {str(error).splitlines()[0]})"
        )
        print(
            "  Start it with: docker compose up -d --wait, and check COCKROACHDB_URL in .env"
        )
        ok = False

    if offline():
        print("Models: offline mode (OFFLINE=1), no API key needed")
        return 0 if ok else 1

    try:
        reply = chat_model().invoke("Reply with the single word: ready").text
        print(
            f"Chat model: OK ({os.getenv('MODEL', DEFAULT_MODEL)} says {reply.strip()[:40]!r})"
        )
    except Exception as error:  # noqa: BLE001
        print(
            f"Chat model: failed ({type(error).__name__}: {str(error).splitlines()[0][:150]})"
        )
        print(
            "  Check MODEL and your API key in .env, or set OFFLINE=1 to start without one"
        )
        ok = False

    try:
        dims = embedding_dims(embeddings())
        print(
            f"Embeddings: OK ({os.getenv('EMBEDDINGS', DEFAULT_EMBEDDINGS)}, {dims} dimensions)"
        )
    except Exception as error:  # noqa: BLE001
        print(
            f"Embeddings: failed ({type(error).__name__}: {str(error).splitlines()[0][:150]})"
        )
        print("  Check EMBEDDINGS and your API key in .env")
        ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
