# AGENTS.md

This project houses various retrieval datasets and tools to make experimentation + training easier.

## Mandatory reading

Before proceeding read docs/*.md to orient to this project. DO NOT PROCEED UNTIL THIS IS DONE.

## Poetry, etc

Note: This python project uses Poetry for dependency management and test execution.

## How to commit

- Stage only relevant files.
- Prepare a commit message with:
  - A short headline.
  - A longer paragraph describing the change.
  - A co-author line for Codex: `Co-authored-by: Codex <codex@openai.com>`.
- Show the full commit message before committing.
- Commits will run a pre-commit hook that runs all tests. If any test fails, the commit is rejected.
- Try to fix code to address test failures

## Testing preferences

- When patching in tests, prefer `unittest.mock.patch` decorators.
- Prefer patch decorators over patch context managers.
- Prefer patch decorators over pytest `monkeypatch` unless there is a specific need.

## Docs / PRDs

docs can be found in docs/ folder
