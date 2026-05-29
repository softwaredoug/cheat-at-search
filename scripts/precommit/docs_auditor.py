#!/usr/bin/env python3
"""Docs auditor pre-commit hook.

Runs opencode to invoke the docs_auditor subagent and check for
documentation drift against the current codebase.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent

# The docs_auditor agent may take a while to read files and analyze.
# We set a generous timeout but don't block commits if it exceeds it.
OPENCODE_TIMEOUT = 180  # seconds


def main() -> int:
    os.chdir(REPO_ROOT)

    opencode = shutil.which("opencode")
    if not opencode:
        print("Warning: opencode not found in PATH, skipping docs audit")
        return 0

    print("Running docs auditor (this may take up to a minute)...")

    # docs_auditor is a subagent, so we invoke the default primary agent
    # with a message that triggers delegation to the subagent.
    cmd = [
        opencode,
        "run",
        "--dangerously-skip-permissions",
        "--format", "default",
        "Run the docs_auditor subagent to audit docs/ and README.md for drift against the current codebase, tests, config, and CLI behavior. Report any blocking issues, warnings, or missing documentation.",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=OPENCODE_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        print(f"Warning: docs auditor timed out after {OPENCODE_TIMEOUT}s, skipping")
        return 0
    except Exception as exc:
        print(f"Warning: docs auditor failed to run: {exc}, skipping")
        return 0

    output = result.stdout + result.stderr
    print(output)

    # Check for blocking issues - handle multiple possible heading formats
    has_blocking_section = (
        "## Blocking issues" in output
        or "## Blocking Issues" in output
        or "### Blocking Issues" in output
        or "### Blocking issues" in output
    )
    has_blocking_items = "DOCS-STALE-" in output

    if has_blocking_section and has_blocking_items:
        print()
        print("ERROR: Docs audit found blocking issues. Please fix documentation drift before committing.")
        return 1

    # Check for stale status - handle both possible formats
    if "Overall status: likely stale" in output or "Overall Docs Health: Likely stale" in output:
        print()
        print("ERROR: Docs audit reports 'likely stale' status. Please review and fix documentation.")
        return 1

    print("Docs audit passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
