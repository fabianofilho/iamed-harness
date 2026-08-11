.PHONY: install run test lint format fixtures demo clean

UV ?= uv

install:
	$(UV) sync --extra dev

# Comando de abertura da sessao: offline, sem API key, alguns segundos.
run:
	$(UV) run harness run mcq_baseline --provider fixture --limit 25

test:
	$(UV) run pytest -q

lint:
	$(UV) run ruff check src tests scripts
	$(UV) run mypy
	$(UV) run mypy --allow-untyped-defs --allow-incomplete-defs --allow-untyped-calls tests scripts

format:
	$(UV) run ruff format src tests scripts
	$(UV) run ruff check --fix src tests scripts

# Regrava os fixtures offline. So e necessario depois de mexer no dataset
# ou no template de prompt.
fixtures:
	$(UV) run python scripts/record_fixtures.py --dataset data/seed_ptbr.jsonl
	$(UV) run harness perturb data/seed_ptbr.jsonl --kind shuffle_choices --out data/seed_ptbr.shuffled.jsonl
	$(UV) run python scripts/record_fixtures.py --dataset data/seed_ptbr.shuffled.jsonl

# Baseline, perturbado, comparacao lado a lado e os dois relatorios HTML.
demo:
	@$(UV) run harness run mcq_baseline --provider fixture
	@$(UV) run harness perturb data/seed_ptbr.jsonl --kind shuffle_choices --out data/seed_ptbr.shuffled.jsonl
	@$(UV) run harness run mcq_perturbed --provider fixture
	@set -e; \
	base=$$(grep -l '"task_name": "mcq_baseline"' runs/*/config.json | sed 's|runs/\(.*\)/config.json|\1|' | sort | tail -1); \
	pert=$$(grep -l '"task_name": "mcq_perturbed"' runs/*/config.json | sed 's|runs/\(.*\)/config.json|\1|' | sort | tail -1); \
	$(UV) run harness compare $$base $$pert; \
	$(UV) run harness report $$base >/dev/null; \
	$(UV) run harness report $$pert --baseline $$base >/dev/null; \
	echo ""; \
	echo "relatorios: runs/$$base/report.html  e  runs/$$pert/report.html"

clean:
	rm -rf .cache .pytest_cache .mypy_cache .ruff_cache
	find runs -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +
