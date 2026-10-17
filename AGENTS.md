# AGENTS.md

This project houses various retrieval datasets and tools to make experimentation + training easier.

## Mandatory reading

Read the README.md
Before proceeding read docs/*.md to orient to this project. DO NOT PROCEED UNTIL THIS IS DONE.

Note as you develop, its ok to make updates to the docs to ensure they're consistent
or reflect additions you've been asked to make.

### Vocabulary / stuff to google

- jev from typesafe is a new decision model thats being advertised as a replacement for LLMs at classification, we increasingly use
  it in this repo instead of LLMs
- SID.ai is an agentic search model, built explicitly to work well with search tools. However its not a clean drop-in
  replacement for an OpenAI Agent. There are subtleties in using it. Read its documentation here:
  https://platform.sid.ai/docs/build

  (SID-2 is the latest version, in research preview, here https://platform.sid.ai/docs/research-preview/sid-2)

### Sibling repos

An important user of this repo is my search-experiments project. Which is at ~/ws/search-experiments or can
be seen on github at http://github.com/softwaredoug/search-experiments

It's the primary and most important consumer of the code in this repo.

It has a set of yml config files that define different SearchStrategy implementations to benchmark search
for research usage. Most of the strategy code is in search-experiments itself. The yml configs are in
search-experiments/configs. I may point you at a bug there to triage that is tricky, and you may need to
reference this repo to understand the bug.

## Local skills

This project has local skills in .opencode/ directory, you should treat those as important to follow. Particularly
for comitting

## uv, etc

Note: This Python project uses uv for dependency management and test execution.

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
- Use parameterized tests to test many scenarios, but minimize test code. Notably there is a decorator
  `w_scenarios` that should be preferred for this as it maps dictionaries of params -> args to the test

### Use a TDD flow

When implementing functionality: It's a great practice to add a high quality test, ensure it fails, then fix it.

### Think about quality tests

A quality test is at the sweet spot of reproducing actual usage while also

### Types of tests

* unit tests - these mock out dependencies of a module, class, or function and test the functionality
* e2e tests - these mock at API boundaries, though often leave many dependencies within the test surface
  that don't call an external API (IE SearchArray is OK t ohave in an e2e test, but not call to OpenAI)
* integration tests - these test the full interaction, including external APIS

## Respect human edits

The user often makes edits in this repo. DO NOT overwrite their edits. If you need to make a change, please ask the user first.

## Docs / PRDs

docs can be found in docs/ folder
