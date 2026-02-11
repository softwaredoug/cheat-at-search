# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase demonstrating LLM-based search ranking improvement techniques. Accompanies the blog article "How Much Does Reasoning Improve Search Quality" by Doug Turnbull.

## Commands

```bash
# Install dependencies
poetry install

# Run tests
poetry run pytest
poetry run pytest tests/test_datasets.py  # single file
poetry run pytest tests/test_datasets.py::test_wands_import -v  # single test

# Run agent experiments
poetry run python -m cheat_at_search.agent.wands search_few_shot  # WANDS dataset
poetry run python -m cheat_at_search.agent.esci  # ESCI dataset

# Type checking
poetry run mypy cheat_at_search/

# Linting
poetry run flake8 cheat_at_search/
```

## Architecture

### Core Abstractions

**SearchStrategy** (`strategy/strategy.py`): Base class for search implementations
- `BM25Search`: Baseline keyword search with Snowball tokenization, title boost (9.3x), description boost (4.1x)
- `ReasoningSearchStrategy`: LLM-based search using agents to reason about results
- `BestPossibleResults`: Oracle showing ideal ranking

**Agent** (`agent/search_client.py`): Abstract base for LLM interaction
- `OpenAIAgent`: OpenAI API with tool calling, returns `SearchResults` Pydantic model
- Tools adapted from Python functions via `make_tool_adapter()` in `pydantize.py`

**EnrichClient** (`enrich/enrich_client.py`): Abstract base for LLM text enrichment
- Implementations: `OpenAIEnricher`, `GoogleEnrichClient`, `OpenRouterEnrichClient`
- `CachedEnrichClient`: Wraps any enricher with persistent disk caching
- `AutoEnricher`: Orchestrator selecting provider, handling batch/serial enrichment with threading

### Data Layer

Dataset loaders (`*_data.py`) provide lazy-loaded access to:
- WANDS (Wayfair furniture)
- ESCI (Amazon shopping)
- MS MARCO (web passages)
- TMDB (movies)

Each exposes: `queries`, `corpus` (documents), `judgments` (relevance labels)

Datasets are git-synced via `sync_git_repo()` in `data_dir.py`.

### Evaluation

- `grade_results()`: Assigns relevance grades to search results
- `idcg_max()`: Computes Ideal DCG for NDCG normalization
- `vs_ideal()` in `search.py`: Side-by-side comparison of actual vs ideal rankings

### Key Data Models

`SearchResults` contains ranked `SearchResult` items with emoji relevance levels: 🤩 (Excellent), 🙂 (Satisfactory), 😐 (Compliant), 😭 (Irrelevant)

Query models in `model/query.py` support progressive enrichment: synonyms, spelling corrections, categories, structured features.

## Configuration

**API Keys**: Set via environment variables (e.g., `OPENAI_API_KEY`) or store in `data/keys.json`. Falls back to interactive prompt.

**Data Path**: Override with `CHEAT_AT_SEARCH_DATA_PATH` env var; defaults to `project_root/data/`.

**Code Style**: Max line length 120 chars. MyPy and Flake8 enabled; Black disabled.

## Commit Convention

Per AGENTS.md: Include co-author line `Co-authored-by: Codex <codex@openai.com>` for AI-assisted commits.
