# Shortcuts. Each target is one or two plain commands, also listed in the README.
.PHONY: setup db check test down

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt
	test -f .env || cp .env.example .env

db:
	docker compose up -d --wait

check:
	.venv/bin/python check_setup.py

test:
	OFFLINE=1 .venv/bin/pytest -q

down:
	docker compose down
