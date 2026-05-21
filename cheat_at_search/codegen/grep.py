from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from cheat_at_search.logger import log_to_stdout


def _resolve_logger(logger=None, logger_name: str = "codegen.grep"):
    if logger is not None:
        return logger
    return log_to_stdout(logger_name=logger_name)


def make_run_path_grep_tool(
    run_path: Path,
    files: dict[str, str] | None = None,
    module_name: str = "rerank_esci",
    logger=None,
) -> Callable[[str, str, int, int], dict]:
    base_path = Path(run_path).expanduser().resolve()
    logger = _resolve_logger(logger)
    files = files or {}
    managed_files = {
        f"{module_name}.py": "reranker source",
        "queries.csv": "training query summary",
    }
    combined_files = {**managed_files, **files}
    files_doc = "\n".join(
        f"- {filename} ({description})"
        for filename, description in combined_files.items()
    )
    if files_doc:
        files_doc = f"\n\nTypical files to inspect:\n{files_doc}"
    training_layout_doc = (
        "\n\nTraining run layout (when logging is enabled):\n"
        "- training/<timestamp>/\n"
        "  - reranker.py\n"
        "  - queries.csv\n"
        "  - <query_path>/results.csv"
    )

    def grep_run_path(
        pattern: str,
        file_glob: str = "**/*",
        max_matches: int = 50,
        max_file_size_kb: int = 512,
    ) -> dict:
        """Search previous codegen run files for a regex pattern.

        Args:
            pattern: Regex pattern to search for.
            file_glob: Glob pattern under the run path to scan.
            max_matches: Maximum number of matches to return.
            max_file_size_kb: Skip files larger than this limit.
        """
        if not base_path.exists():
            return {"matches": [], "error": f"run path not found: {base_path}"}

        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return {"matches": [], "error": f"invalid regex: {exc}"}
        logger.info(
            "!GREP Searching for pattern '%s' in files matching '%s' under %s...",
            pattern,
            file_glob,
            base_path,
        )

        matches = []
        skipped = []
        truncated = False
        for path in sorted(base_path.rglob(file_glob)):
            if len(matches) >= max_matches:
                truncated = True
                break
            if path.is_dir():
                continue
            try:
                size_kb = path.stat().st_size / 1024
            except OSError:
                skipped.append(str(path))
                continue
            if size_kb > max_file_size_kb:
                skipped.append(
                    f"{path} (size {size_kb:.1f}kb > {max_file_size_kb}kb)"
                )
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                skipped.append(f"{path} (non-utf8)")
                continue
            except OSError:
                skipped.append(str(path))
                continue
            for line_num, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    matches.append({"file": str(path), "line": line_num, "text": line})
                    if len(matches) >= max_matches:
                        truncated = True
                        break
        if matches:
            logger.info("!GREP Found %d matches", len(matches))
        else:
            logger.info("!GREP No matches found")
        return {"matches": matches, "truncated": truncated, "skipped": skipped}

    grep_run_path.__name__ = "grep_run_path"
    grep_run_path.__doc__ = f"""Search previous codegen run files for a regex pattern.{files_doc}{training_layout_doc}

    Args:
        pattern: Regex pattern to search for.
        file_glob: Glob pattern under the run path to scan.
        max_matches: Maximum number of matches to return.
        max_file_size_kb: Skip files larger than this limit.
    """
    return grep_run_path
