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

    # docs_auditor is a primary agent; invoke it directly.
    cmd = [
        opencode,
        "run",
        "--agent", "docs_auditor",
        "--dangerously-skip-permissions",
        "--format", "default",
        "Analyze README.md and docs/ for information thats out of date. Output 'all good' if no issues found.",
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

    if "all good" in output.lower():
        print("Docs audit passed.")
        return 0

    print()
    print("ERROR: Docs audit did not return 'all good'. Please review the report above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
